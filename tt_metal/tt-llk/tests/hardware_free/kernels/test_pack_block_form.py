# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Python mirror of the form llk_pack_block takes per cell (tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_tile_api.h), checked against the header.

A block takes the per-tile pack (main's code, T), the plain or closed block that programs the output (B) or the block with the W offset (W). The form tables hold one row per output
class (tiles of 16x16, 16x32, 8x32, 4x32 and 1x32, Bfp8_b and Bfp8, Bfp4_b and Bfp2_b, Bfp4 and Bfp2, 4 byte datums, the other plain formats) and one column per DEST width and sync, from the tt-llk crossover sweeps.
The first block of an output takes the init or the reconfigure table, the blocks that continue it the next table. Behind matmul math the cells the matmul sweep read slower are forbidden (masks below).
"""

import pathlib
import re

API = pathlib.Path(__file__).resolve().parents[4] / "hw/ckernels/blackhole/metal/llk_api/llk_pack_tile_api.h"
ALT = pathlib.Path(__file__).resolve().parents[2] / "sources/pack_block_alt_test.cpp"
CLASSES = ("s16x16", "s16x32", "s8x32", "s4x32", "s1x32", "bfp8", "bfp4", "bfpa", "wide", "narrow")

INIT_FORM = {
    "s16x16": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s16x32": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s8x32": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s4x32": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s1x32": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "bfp8": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB"],
    "bfp4": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB"],
    "bfpa": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TTBTBTTTTTTTTTTT"],
    "wide": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "narrow": ["TBBBBBBBBBBBBBBB", "TBBTTTTTTTTTTTTT", "BBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
}
RECONF_FORM = {
    "s16x16": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s16x32": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s8x32": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s4x32": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s1x32": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "bfp8": ["TBBBBBBBBBBBBBBB", "TBTBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB"],
    "bfp4": ["TBBBBBBBBBBBBBBB", "TBBTTTTTTTTTTTTT", "TBBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB"],
    "bfpa": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "wide": ["BBBBBBBBBBBBBBBB", "BBTTTTTTTTTTTTTT", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB"],
    "narrow": ["BBBBBBBBBBBBBBBB", "BBTTTTTTTTTTTTTT", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB"],
}
NEXT_FORM = {
    "s16x16": ["TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBBWBBBBBBBB", "TWWBBBBBBBBBBBBB"],
    "s16x32": ["TWWWBWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBBBBBBWWWWW", "TWWBBBBBBBBBBBBB"],
    "s8x32": ["TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBWWBBBBBBBB", "TWWBBBBBBBBBBBBB"],
    "s4x32": ["TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBWWBBBBBBBB", "TWWBBBBBBBBBBBBB"],
    "s1x32": ["TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBWWBBBBBBBB", "TWWBBBBWWWWWWWWW"],
    "bfp8": ["TWBBBBWWWWWWWWWW", "TBWTTTTTTTTTTTTT", "TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW"],
    "bfp4": ["TWBBWWWWWWWWWWWW", "TBBBBBBBBBBBBBBB", "TWWWWWWWWWWWWWWW", "TBWWWWWWWWWWWWWW"],
    "bfpa": ["TWBBWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW"],
    "wide": ["BWWWBBBBBBBBBBBB", "TBWBBBBBBBBBBBBB", "BWWWWWWWWWWWWWWW", "WWWWWWWWWWWWWWWW"],
    "narrow": ["BBBWWWWWWWWWWWWW", "TBBTTTTTTTTTTTTT", "BBBWWWWWWWWWWWWW", "WWWWWWWWWWWWWWWW"],
}
# Matmul overlay: bit n-1 forbids the block (NO_BLOCK) or the W offset (NO_W) at n tiles. [group][class][column]: groups LoFi K not given, LoFi K 1, LoFi K above 1, HiFi2 to HiFi4;
# classes Bfp8_b to Bfp2_b and Bfp8, Bfp4 and Bfp2, other plain formats, 4 byte datums.
MM_NO_BLOCK = [[[168, 8, 0, 0], [232, 0, 0, 0], [160, 0, 0, 0], [128, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[168, 8, 0, 0], [232, 0, 0, 0], [160, 0, 0, 0], [128, 0, 0, 0]], [[98, 2, 0, 0], [34, 0, 0, 0], [0, 0, 0, 0], [64, 0, 0, 0]]]
MM_NO_W = [[[184, 8, 10, 2], [232, 0, 2, 0], [160, 0, 0, 0], [128, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[184, 8, 10, 2], [232, 0, 2, 0], [160, 0, 0, 0], [128, 0, 0, 0]], [[98, 2, 0, 0], [34, 0, 0, 0], [2, 2, 0, 0], [64, 0, 0, 0]]]
OVERLAY_CLASS = {"bfp8": 0, "bfp4": 0, "bfpa": 1, "narrow": 2, "wide": 3}
FORM_NAMES = {"T": "per_tile", "B": "block", "W": "w_offset"}
BFP_8 = ("Bfp8_b", "Bfp8")
BFP_4 = ("Bfp4_b", "Bfp2_b")
BFP_A = ("Bfp4", "Bfp2")
WIDE = ("Float32", "Int32", "UInt32", "Tf32")


def output_class(dst_format, num_faces=4, face_r_dim=16):
    if num_faces != 4 or face_r_dim != 16:
        if face_r_dim == 16:
            return "s16x16" if num_faces == 1 else "s16x32"
        return "s8x32" if face_r_dim >= 8 else ("s4x32" if face_r_dim >= 4 else "s1x32")
    if dst_format in BFP_A:
        return "bfpa"
    if dst_format in BFP_8:
        return "bfp8"
    if dst_format in BFP_4:
        return "bfp4"
    return "wide" if dst_format in WIDE else "narrow"


def pack_block_form(ntiles, dst_format, state, fp32_dest=False, full_sync=False, hifi=False, num_faces=4, face_r_dim=16, matmul_kt=0):
    """Form of one llk_pack_block call; state is init (first block after a pack init or another output), reconf (first block after a packer reconfigure) or next (the block continues the output)."""
    table = {"init": INIT_FORM, "reconf": RECONF_FORM, "next": NEXT_FORM}[state]
    column = (2 if full_sync else 0) + (1 if fp32_dest else 0)
    cls = output_class(dst_format, num_faces, face_r_dim)
    count = min(ntiles, 16) - 1
    form = table[cls][column][count]
    if form != "T" and cls in OVERLAY_CLASS:
        group = 3 if hifi else (0 if matmul_kt == 0 else (1 if matmul_kt == 1 else 2))
        if (MM_NO_BLOCK[group][OVERLAY_CLASS[cls]][column] >> count) & 1:
            form = "T"
        elif form == "W" and (MM_NO_W[group][OVERLAY_CLASS[cls]][column] >> count) & 1:
            form = "B"
    return FORM_NAMES[form]


def test_mirror_cells():
    assert pack_block_form(1, 'Float16_b', 'init') == 'per_tile'
    assert pack_block_form(2, 'Float16_b', 'init') == 'block'
    assert pack_block_form(2, 'Float16_b', 'next') == 'block'
    assert pack_block_form(4, 'Float16_b', 'next', fp32_dest=True) == 'per_tile'
    assert pack_block_form(2, 'Bfp8_b', 'reconf') == 'block'
    assert pack_block_form(3, 'Bfp8_b', 'init') == 'block'
    assert pack_block_form(8, 'Bfp8_b', 'next') == 'per_tile'
    assert pack_block_form(12, 'Bfp8_b', 'next', full_sync=True) == 'w_offset'
    assert pack_block_form(16, 'Bfp8_b', 'init', full_sync=True) == 'block'
    assert pack_block_form(12, 'Float32', 'next', full_sync=True) == 'w_offset'
    assert pack_block_form(16, 'Float16_b', 'reconf', full_sync=True) == 'block'
    assert pack_block_form(4, 'Int8', 'init') == 'block'
    assert pack_block_form(16, 'Fp8_e4m3', 'next', full_sync=True) == 'w_offset'
    assert pack_block_form(3, 'Tf32', 'next') == 'w_offset'
    assert pack_block_form(4, 'Bfp8_b', 'next', matmul_kt=4) == 'per_tile'
    assert pack_block_form(4, 'Bfp8_b', 'next', matmul_kt=1) == 'block'
    assert pack_block_form(4, 'Bfp8_b', 'next', hifi=True) == 'block'
    assert pack_block_form(4, 'Bfp8_b', 'next') == 'per_tile'
    assert pack_block_form(3, 'Float16_b', 'init', num_faces=2, face_r_dim=16) == 'block'
    assert pack_block_form(5, 'Float16_b', 'next', num_faces=2, face_r_dim=8) == 'w_offset'


def _table(src, name):
    m = re.search(rf"constexpr char {name}\[10\]\[4\]\[17\] = \{{(.*?)\n\}};", src, re.S)
    assert m, name
    rows = re.findall(r"\{((?:\"[TBW]{16}\",? ?){4})\},\s*//\s*(.*)", m.group(1))
    return {CLASSES[i]: re.findall(r"\"([TBW]{16})\"", r) for i, (r, _) in enumerate(rows)}


def _masks(src, name):
    m = re.search(rf"constexpr std::uint16_t {name}\[4\]\[4\]\[4\] = \{{(.*?)\n\}};", src, re.S)
    assert m, name
    return [[[int(x, 16) for x in re.findall(r"0x([0-9A-F]{4})", cls)] for cls in re.findall(r"\{(0x[^}]*)\}", g)] for g in re.findall(r"\{(\{.*\})\},", m.group(1))]


def test_header_tables_match_mirror():
    src = API.read_text()
    assert _table(src, "llk_pack_init_form") == INIT_FORM
    assert _table(src, "llk_pack_reconf_form") == RECONF_FORM
    assert _table(src, "llk_pack_next_form") == NEXT_FORM
    assert _masks(src, "llk_pack_mm_no_block") == MM_NO_BLOCK
    assert _masks(src, "llk_pack_mm_no_w") == MM_NO_W


def test_header_logic_matches_mirror():
    src = " ".join(API.read_text().split())
    assert "const std::uint32_t column = (DST_SYNC_MODE == DstSync::SyncFull ? 2 : 0) + (is_fp32_dest_acc_en ? 1 : 0);" in src
    assert "const std::uint32_t count = (ntiles < 16 ? ntiles : 16) - 1;" in src
    assert "constexpr std::uint32_t group = MATH_FIDELITY != MathFidelity::LoFi ? 3 : (matmul_kt == 0 ? 0 : (matmul_kt == 1 ? 1 : 2));" in src
    assert "constexpr std::uint32_t overlay_row[5] = {0, 0, 1, 3, 2};" in src
    assert "if (form == 'T' || row < 5) { return form; }" in src
    assert "if (get_output_num_faces(output_id) != 4 || rows != FACE_R_DIM) { if (rows == FACE_R_DIM) { return get_output_num_faces(output_id) == 1 ? 0 : 1; } return rows >= 8 ? 2 : (rows >= 4 ? 3 : 4); }" in src
    assert "return llk_pack_dst_is_wide(output_id) ? 8 : 9;" in src
    assert "first_block ? (pack_reconfigured ? llk_pack_reconf_form : llk_pack_init_form) : llk_pack_next_form" in src
    assert "closed_pays" not in src and "closed_loses" not in src and "LLK_PACK_BLOCK_MIN_TILES" not in src


def test_alt_kernel_mirrors_reconf_rows():
    """pack_block_alt_test.cpp holds the reconfigure rows of its three formats (Bfp8_b, Float32, Float16_b) as llk_pack_block takes them with the matmul K not given."""
    src = ALT.read_text()
    m = re.search(r"constexpr char alt_reconf_form\[3\]\[4\]\[17\] = \{(.*?)\n\};", src, re.S)
    rows = [re.findall(r"\"([TBW]{16})\"", r) for r in re.findall(r"\{((?:\"[TBW]{16}\",? ?){4})\}", m.group(1))]
    expect = [[("".join(_alt(c, col, n) for n in range(16))) for col in range(4)] for c in ("bfp8", "wide", "narrow")]
    assert rows == expect


def _alt(cls, col, count):
    letter = RECONF_FORM[cls][col][count]
    if letter != "T" and (MM_NO_BLOCK[0][OVERLAY_CLASS[cls]][col] >> count) & 1:
        return "T"
    return letter
