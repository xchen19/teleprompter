#!/usr/bin/env python3
"""
Generate a minimal valid MP4 (H.264/AVC + AAC audio) video file.
- 320x90 black pixels (encoded as 320x96 with bottom crop), single IDR frame.
- Silent AAC-LC mono audio track (critical for iOS PiP to persist across apps).
- Duration set to 3600 seconds in headers.
- Compatible with iOS Safari.
- Uses only Python standard library.
"""

import struct
import os

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blank.mp4")

# ── helpers ──────────────────────────────────────────────────────────────────

def box(box_type: bytes, payload: bytes) -> bytes:
    """Wrap payload in an MP4 box (atom)."""
    return struct.pack(">I", 8 + len(payload)) + box_type + payload


def full_box(box_type: bytes, version: int, flags: int, payload: bytes) -> bytes:
    """Wrap payload in an MP4 full-box (version + flags header)."""
    vf = struct.pack(">I", (version << 24) | (flags & 0x00FFFFFF))
    return box(box_type, vf + payload)


def _exp_golomb(val: int) -> str:
    """Return exp-Golomb coded bits for *val* as a bitstring."""
    if val == 0:
        return "1"
    code = val + 1
    bits = bin(code)[2:]
    return "0" * (len(bits) - 1) + bits


def _bits_to_bytes(bitstring: str) -> bytes:
    """Convert a bitstring to bytes, padding with trailing zeros."""
    while len(bitstring) % 8 != 0:
        bitstring += "0"
    out = bytearray()
    for i in range(0, len(bitstring), 8):
        out.append(int(bitstring[i:i + 8], 2))
    return bytes(out)


def nal_length_prefixed(nal: bytes) -> bytes:
    """Prefix a NAL unit with its 4-byte big-endian length."""
    return struct.pack(">I", len(nal)) + nal


def apply_emulation_prevention(rbsp: bytes) -> bytes:
    """Insert H.264 emulation prevention bytes (0x03) into RBSP."""
    result = bytearray()
    zeros = 0
    for b in rbsp:
        if zeros >= 2 and b <= 3:
            result.append(0x03)
            zeros = 0
        result.append(b)
        zeros = zeros + 1 if b == 0 else 0
    return bytes(result)


def _desc_size(size: int) -> bytes:
    """Encode MPEG-4 descriptor length in expandable 4-byte format."""
    return bytes([
        0x80 | ((size >> 21) & 0x7F),
        0x80 | ((size >> 14) & 0x7F),
        0x80 | ((size >> 7) & 0x7F),
        size & 0x7F,
    ])


# ── constants ────────────────────────────────────────────────────────────────

# Video: 320x90 display, 320x96 coded (20x6 macroblocks)
WIDTH_DISPLAY = 320
HEIGHT_DISPLAY = 90
MB_W = 20   # 320 / 16
MB_H = 6    # 96  / 16
HEIGHT_CODED = MB_H * 16          # 96
CROP_BOTTOM = (HEIGHT_CODED - HEIGHT_DISPLAY) // 2  # 3 chroma units for 4:2:0
NUM_MBS = MB_W * MB_H             # 120

# Timing
TIMESCALE = 600                   # movie timescale
DURATION_S = 3600
DURATION_TICKS = TIMESCALE * DURATION_S  # 2,160,000

# Audio: AAC-LC mono 8000 Hz
AUDIO_SR = 8000
AUDIO_DURATION_TICKS = AUDIO_SR * DURATION_S  # 28,800,000

# AudioSpecificConfig: AAC-LC(2), 8000Hz(idx 11), mono(1)
# Bits: 00010 1011 0001 000  =>  0x15 0x88
AUDIO_SPECIFIC_CONFIG = bytes([0x15, 0x88])

# ── H.264 NAL unit builders ─────────────────────────────────────────────────

def build_sps() -> bytes:
    """SPS for 320x96 Baseline profile, cropped to 320x90."""
    b = ""
    b += "01000010"              # profile_idc = 66 (Baseline)
    b += "11000000"              # constraint flags
    b += format(21, '08b')       # level_idc = 21 (Level 2.1)
    b += _exp_golomb(0)          # seq_parameter_set_id
    b += _exp_golomb(0)          # log2_max_frame_num_minus4
    b += _exp_golomb(0)          # pic_order_cnt_type
    b += _exp_golomb(0)          # log2_max_pic_order_cnt_lsb_minus4
    b += _exp_golomb(0)          # max_num_ref_frames
    b += "0"                     # gaps_in_frame_num_value_allowed_flag
    b += _exp_golomb(MB_W - 1)   # pic_width_in_mbs_minus1 = 19
    b += _exp_golomb(MB_H - 1)   # pic_height_in_map_units_minus1 = 5
    b += "1"                     # frame_mbs_only_flag
    b += "0"                     # direct_8x8_inference_flag
    b += "1"                     # frame_cropping_flag
    b += _exp_golomb(0)          # crop_left
    b += _exp_golomb(0)          # crop_right
    b += _exp_golomb(0)          # crop_top
    b += _exp_golomb(CROP_BOTTOM)  # crop_bottom = 3 => 96 - 2*3 = 90
    b += "0"                     # vui_parameters_present_flag
    b += "1"                     # rbsp_stop_one_bit
    rbsp = _bits_to_bytes(b)
    return b"\x67" + apply_emulation_prevention(rbsp)


def build_pps() -> bytes:
    """Minimal PPS for CAVLC baseline."""
    b = ""
    b += _exp_golomb(0)   # pic_parameter_set_id
    b += _exp_golomb(0)   # seq_parameter_set_id
    b += "0"              # entropy_coding_mode_flag (CAVLC)
    b += "0"              # bottom_field_pic_order_in_frame_present_flag
    b += _exp_golomb(0)   # num_slice_groups_minus1
    b += _exp_golomb(0)   # num_ref_idx_l0_default_active_minus1
    b += _exp_golomb(0)   # num_ref_idx_l1_default_active_minus1
    b += "0"              # weighted_pred_flag
    b += "00"             # weighted_bipred_idc
    b += _exp_golomb(0)   # pic_init_qp_minus26
    b += _exp_golomb(0)   # pic_init_qs_minus26
    b += _exp_golomb(0)   # chroma_qp_index_offset
    b += "0"              # deblocking_filter_control_present_flag
    b += "0"              # constrained_intra_pred_flag
    b += "0"              # redundant_pic_cnt_present_flag
    b += "1"              # rbsp_stop_one_bit
    rbsp = _bits_to_bytes(b)
    return b"\x68" + apply_emulation_prevention(rbsp)


def build_idr_slice() -> bytes:
    """IDR slice with 120 all-black I_PCM macroblocks."""
    # -- slice header --
    hdr = ""
    hdr += _exp_golomb(0)   # first_mb_in_slice
    hdr += _exp_golomb(7)   # slice_type = 7 (I, all pictures)
    hdr += _exp_golomb(0)   # pic_parameter_set_id
    hdr += "0000"           # frame_num (4 bits, log2_max=4)
    hdr += _exp_golomb(0)   # idr_pic_id
    hdr += "0000"           # pic_order_cnt_lsb (4 bits)
    hdr += "0"              # no_output_of_prior_pics_flag
    hdr += "0"              # long_term_reference_flag
    hdr += _exp_golomb(0)   # slice_qp_delta

    # -- macroblock PCM data (same for every MB) --
    pcm = bytearray(384)    # 256 luma + 64 Cb + 64 Cr
    # Luma: 0 (black) -- already zero
    # Chroma Cb + Cr: 128 (neutral)
    for i in range(256, 384):
        pcm[i] = 128
    pcm_bytes = bytes(pcm)

    # Build RBSP in chunks to avoid huge bitstrings
    parts = []
    bits = hdr
    for _ in range(NUM_MBS):
        bits += _exp_golomb(25)          # mb_type = I_PCM
        while len(bits) % 8 != 0:
            bits += "1"                  # pcm_alignment_one_bit
        parts.append(_bits_to_bytes(bits))
        bits = ""
        parts.append(pcm_bytes)

    # RBSP stop bit (byte-aligned here, so 1 + 7 zeros = 0x80)
    parts.append(b"\x80")

    rbsp = b"".join(parts)
    ebsp = apply_emulation_prevention(rbsp)
    return b"\x65" + ebsp


# ── silent AAC frame ────────────────────────────────────────────────────────

def build_silent_aac_frame() -> bytes:
    """
    Minimal raw AAC-LC silent frame for mono, 1024 samples.
    Layout (31 bits, padded to 4 bytes):
      000        ID_SCE (3 bits)
      0000       element_instance_tag (4 bits)
      00000000   global_gain (8 bits)
      0          ics_reserved_bit (1 bit)
      00         window_sequence = ONLY_LONG_SEQUENCE (2 bits)
      0          window_shape (1 bit)
      000000     max_sfb = 0 (6 bits) -- no spectral bands
      0          predictor_data_present (1 bit)
      0          pulse_data_present (1 bit)
      0          tns_data_present (1 bit)
      111        ID_END (3 bits)
      0          padding (1 bit)
    """
    return bytes([0x00, 0x00, 0x00, 0x0E])


# ── MP4 container boxes ─────────────────────────────────────────────────────

def build_ftyp() -> bytes:
    p = b"isom"                                  # major brand
    p += struct.pack(">I", 0x200)                # minor version
    p += b"isom" + b"iso2" + b"avc1" + b"mp41"   # compatible brands
    return box(b"ftyp", p)


def build_mvhd() -> bytes:
    d = b""
    d += struct.pack(">I", 0)               # creation_time
    d += struct.pack(">I", 0)               # modification_time
    d += struct.pack(">I", TIMESCALE)       # timescale
    d += struct.pack(">I", DURATION_TICKS)  # duration
    d += struct.pack(">I", 0x00010000)      # rate 1.0
    d += struct.pack(">H", 0x0100)          # volume 1.0
    d += b"\x00" * 10                       # reserved
    for v in (0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000):
        d += struct.pack(">I", v)
    d += b"\x00" * 24                       # pre_defined
    d += struct.pack(">I", 3)               # next_track_ID (we have 2 tracks)
    return full_box(b"mvhd", 0, 0, d)


def build_dinf() -> bytes:
    """Data information box with self-contained data reference."""
    dref_entry = full_box(b"url ", 0, 1, b"")
    dref_data = struct.pack(">I", 1) + dref_entry
    return box(b"dinf", full_box(b"dref", 0, 0, dref_data))


# ── Video track ──────────────────────────────────────────────────────────────

def build_video_tkhd() -> bytes:
    d = b""
    d += struct.pack(">I", 0)                          # creation_time
    d += struct.pack(">I", 0)                          # modification_time
    d += struct.pack(">I", 1)                          # track_ID
    d += struct.pack(">I", 0)                          # reserved
    d += struct.pack(">I", DURATION_TICKS)             # duration
    d += b"\x00" * 8                                   # reserved
    d += struct.pack(">H", 0)                          # layer
    d += struct.pack(">H", 0)                          # alternate_group
    d += struct.pack(">H", 0)                          # volume (0 for video)
    d += b"\x00" * 2                                   # reserved
    for v in (0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000):
        d += struct.pack(">I", v)
    d += struct.pack(">I", WIDTH_DISPLAY << 16)        # width  320.0 (16.16)
    d += struct.pack(">I", HEIGHT_DISPLAY << 16)       # height  90.0 (16.16)
    return full_box(b"tkhd", 0, 3, d)  # flags = enabled | in_movie


def build_video_mdhd() -> bytes:
    d = b""
    d += struct.pack(">I", 0)
    d += struct.pack(">I", 0)
    d += struct.pack(">I", TIMESCALE)
    d += struct.pack(">I", DURATION_TICKS)
    d += struct.pack(">H", 0x55C4)  # language 'und'
    d += struct.pack(">H", 0)
    return full_box(b"mdhd", 0, 0, d)


def build_video_hdlr() -> bytes:
    d = struct.pack(">I", 0) + b"vide" + b"\x00" * 12 + b"VideoHandler\x00"
    return full_box(b"hdlr", 0, 0, d)


def build_vmhd() -> bytes:
    d = struct.pack(">H", 0) + struct.pack(">HHH", 0, 0, 0)
    return full_box(b"vmhd", 0, 1, d)  # flags = 1 required by spec


def build_video_stbl(sample_data: bytes, sps: bytes, pps: bytes) -> bytes:
    # -- avcC (AVC Decoder Configuration Record) --
    avcc_d = b""
    avcc_d += struct.pack("B", 1)           # configurationVersion
    avcc_d += sps[1:4]                      # profile, compat, level
    avcc_d += struct.pack("B", 0xFF)        # lengthSizeMinusOne = 3 | reserved
    avcc_d += struct.pack("B", 0xE1)        # numSPS = 1 | reserved
    avcc_d += struct.pack(">H", len(sps))
    avcc_d += sps
    avcc_d += struct.pack("B", 1)           # numPPS
    avcc_d += struct.pack(">H", len(pps))
    avcc_d += pps
    avcc = box(b"avcC", avcc_d)

    # -- avc1 sample entry --
    avc1_d = b""
    avc1_d += b"\x00" * 6                          # reserved
    avc1_d += struct.pack(">H", 1)                 # data_reference_index
    avc1_d += b"\x00" * 16                          # pre_defined + reserved
    avc1_d += struct.pack(">H", WIDTH_DISPLAY)      # width
    avc1_d += struct.pack(">H", HEIGHT_DISPLAY)     # height
    avc1_d += struct.pack(">I", 0x00480000)         # horiz res 72 dpi
    avc1_d += struct.pack(">I", 0x00480000)         # vert res 72 dpi
    avc1_d += struct.pack(">I", 0)                  # reserved
    avc1_d += struct.pack(">H", 1)                  # frame_count
    avc1_d += b"\x00" * 32                          # compressorname
    avc1_d += struct.pack(">H", 0x0018)             # depth = 24
    avc1_d += struct.pack(">h", -1)                 # pre_defined
    avc1_d += avcc
    avc1 = box(b"avc1", avc1_d)

    stsd = full_box(b"stsd", 0, 0, struct.pack(">I", 1) + avc1)

    # 1 sample with duration = entire movie
    stts = full_box(b"stts", 0, 0, struct.pack(">III", 1, 1, DURATION_TICKS))

    # sync sample table -- sample 1 is a keyframe
    stss = full_box(b"stss", 0, 0, struct.pack(">II", 1, 1))

    # sample-to-chunk
    stsc = full_box(b"stsc", 0, 0, struct.pack(">IIII", 1, 1, 1, 1))

    # sample size
    stsz = full_box(b"stsz", 0, 0, struct.pack(">III", 0, 1, len(sample_data)))

    # chunk offset -- placeholder, patched later
    stco = full_box(b"stco", 0, 0, struct.pack(">II", 1, 0))

    return box(b"stbl", stsd + stts + stss + stsc + stsz + stco)


# ── Audio track ──────────────────────────────────────────────────────────────

def build_audio_tkhd() -> bytes:
    d = b""
    d += struct.pack(">I", 0)               # creation_time
    d += struct.pack(">I", 0)               # modification_time
    d += struct.pack(">I", 2)               # track_ID = 2
    d += struct.pack(">I", 0)               # reserved
    d += struct.pack(">I", DURATION_TICKS)  # duration (in movie timescale)
    d += b"\x00" * 8                        # reserved
    d += struct.pack(">H", 0)               # layer
    d += struct.pack(">H", 0)               # alternate_group
    d += struct.pack(">H", 0x0100)          # volume = 1.0 (audio)
    d += b"\x00" * 2                        # reserved
    for v in (0x00010000, 0, 0, 0, 0x00010000, 0, 0, 0, 0x40000000):
        d += struct.pack(">I", v)
    d += struct.pack(">I", 0)               # width = 0 (audio)
    d += struct.pack(">I", 0)               # height = 0 (audio)
    return full_box(b"tkhd", 0, 3, d)


def build_audio_mdhd() -> bytes:
    d = b""
    d += struct.pack(">I", 0)
    d += struct.pack(">I", 0)
    d += struct.pack(">I", AUDIO_SR)              # timescale = sample rate
    d += struct.pack(">I", AUDIO_DURATION_TICKS)  # duration in audio timescale
    d += struct.pack(">H", 0x55C4)                # language 'und'
    d += struct.pack(">H", 0)
    return full_box(b"mdhd", 0, 0, d)


def build_audio_hdlr() -> bytes:
    d = struct.pack(">I", 0) + b"soun" + b"\x00" * 12 + b"SoundHandler\x00"
    return full_box(b"hdlr", 0, 0, d)


def build_smhd() -> bytes:
    d = struct.pack(">HH", 0, 0)  # balance + reserved
    return full_box(b"smhd", 0, 0, d)


def build_esds() -> bytes:
    """Build esds box with AAC-LC decoder configuration."""
    asc = AUDIO_SPECIFIC_CONFIG  # 2 bytes

    # DecoderSpecificInfo descriptor (tag 0x05)
    dsi = bytes([0x05]) + _desc_size(len(asc)) + asc

    # DecoderConfigDescriptor (tag 0x04)
    dcd_payload = b""
    dcd_payload += struct.pack("B", 0x40)    # objectTypeIndication = MPEG-4 Audio
    dcd_payload += struct.pack("B", 0x15)    # streamType=5(audio)<<2 | upStream=0 | reserved=1
    dcd_payload += b"\x00\x00\x00"           # bufferSizeDB
    dcd_payload += struct.pack(">I", 0)      # maxBitrate
    dcd_payload += struct.pack(">I", 0)      # avgBitrate
    dcd_payload += dsi
    dcd = bytes([0x04]) + _desc_size(len(dcd_payload)) + dcd_payload

    # SL_ConfigDescriptor (tag 0x06)
    sld = bytes([0x06]) + _desc_size(1) + bytes([0x02])  # predefined = MP4

    # ES_Descriptor (tag 0x03)
    esd_payload = b""
    esd_payload += struct.pack(">H", 2)      # ES_ID
    esd_payload += struct.pack("B", 0)        # flags + streamPriority
    esd_payload += dcd
    esd_payload += sld
    esd = bytes([0x03]) + _desc_size(len(esd_payload)) + esd_payload

    return full_box(b"esds", 0, 0, esd)


def build_audio_stbl(audio_sample: bytes) -> bytes:
    # -- mp4a sample entry --
    esds = build_esds()

    mp4a_d = b""
    mp4a_d += b"\x00" * 6                           # reserved
    mp4a_d += struct.pack(">H", 1)                  # data_reference_index
    mp4a_d += b"\x00" * 8                            # reserved (version, revision, vendor)
    mp4a_d += struct.pack(">H", 1)                   # channelcount = mono
    mp4a_d += struct.pack(">H", 16)                  # samplesize = 16 bits
    mp4a_d += struct.pack(">H", 0)                   # compression_id
    mp4a_d += struct.pack(">H", 0)                   # packet_size
    mp4a_d += struct.pack(">I", AUDIO_SR << 16)      # samplerate (16.16 fixed-point)
    mp4a_d += esds
    mp4a = box(b"mp4a", mp4a_d)

    stsd = full_box(b"stsd", 0, 0, struct.pack(">I", 1) + mp4a)

    # 1 sample declaring full audio duration
    stts = full_box(b"stts", 0, 0,
                    struct.pack(">III", 1, 1, AUDIO_DURATION_TICKS))

    # No stss for audio (all samples implicitly sync)

    stsc = full_box(b"stsc", 0, 0, struct.pack(">IIII", 1, 1, 1, 1))

    stsz = full_box(b"stsz", 0, 0,
                    struct.pack(">III", 0, 1, len(audio_sample)))

    # chunk offset -- placeholder, patched later
    stco = full_box(b"stco", 0, 0, struct.pack(">II", 1, 0))

    return box(b"stbl", stsd + stts + stsc + stsz + stco)


# ── assembly ─────────────────────────────────────────────────────────────────

def build_mp4():
    """Assemble and write the complete MP4 file."""
    sps = build_sps()
    pps = build_pps()
    idr = build_idr_slice()

    video_sample = nal_length_prefixed(idr)
    audio_sample = build_silent_aac_frame()

    # -- Video track --
    v_stbl = build_video_stbl(video_sample, sps, pps)
    v_minf = box(b"minf", build_vmhd() + build_dinf() + v_stbl)
    v_mdia = box(b"mdia", build_video_mdhd() + build_video_hdlr() + v_minf)
    v_trak = box(b"trak", build_video_tkhd() + v_mdia)

    # -- Audio track --
    a_stbl = build_audio_stbl(audio_sample)
    a_minf = box(b"minf", build_smhd() + build_dinf() + a_stbl)
    a_mdia = box(b"mdia", build_audio_mdhd() + build_audio_hdlr() + a_minf)
    a_trak = box(b"trak", build_audio_tkhd() + a_mdia)

    # -- Top-level boxes --
    moov = box(b"moov", build_mvhd() + v_trak + a_trak)
    ftyp = build_ftyp()
    mdat = box(b"mdat", video_sample + audio_sample)

    file_ba = bytearray(ftyp + moov + mdat)

    # -- Patch stco offsets --
    # Video data starts at mdat payload (after 8-byte mdat header)
    video_offset = len(ftyp) + len(moov) + 8
    # Audio data follows video data in mdat
    audio_offset = video_offset + len(video_sample)

    # Find both stco boxes (first = video track, second = audio track)
    stco_positions = []
    pos = 0
    while True:
        pos = file_ba.find(b"stco", pos)
        if pos == -1:
            break
        stco_positions.append(pos)
        pos += 4

    if len(stco_positions) != 2:
        raise RuntimeError(f"Expected 2 stco boxes, found {len(stco_positions)}")

    # Each stco: type(4) + version_flags(4) + entry_count(4) + offset(4)
    # stco_pos points to 'stco' text; offset field is at stco_pos + 12
    struct.pack_into(">I", file_ba, stco_positions[0] + 12, video_offset)
    struct.pack_into(">I", file_ba, stco_positions[1] + 12, audio_offset)

    with open(OUTPUT_PATH, "wb") as f:
        f.write(file_ba)

    size = os.path.getsize(OUTPUT_PATH)
    print(f"Written {OUTPUT_PATH} ({size} bytes)")
    return OUTPUT_PATH


if __name__ == "__main__":
    build_mp4()
