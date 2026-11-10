# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Python mirror of the form llk_pack_block takes per cell (tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_tile_api.h), checked against the header.

A block takes the per-tile pack (main's code, T), the plain or closed block that programs the output (B) or the block with the W offset (W). The form tables hold one row per output
class (tiles of 16x16, 16x32, 8x32, 4x32 and 1x32, Bfp8_b and Bfp8, Bfp4_b and Bfp2_b, Bfp4 and Bfp2, 4 byte datums, the other plain formats) and one column per DEST width and sync, from the tt-llk crossover sweeps.
The first block of an output takes the init or the reconfigure table, the blocks that continue it the next table. A kernel with a 32 bit CB next to a 32 bit DEST takes the tables of all sweeps, the others the _DC tables of the datacopy
family. Behind matmul math the cells the matmul sweeps read slower are forbidden and, with the K given, the cells where the W offset reads faster take it (masks below).
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
INIT_FORM_DC = {
    "s16x16": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s16x32": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s8x32": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s4x32": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s1x32": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "bfp8": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB"],
    "bfp4": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB"],
    "bfpa": ["TTBBBBBBBBBBBBBB", "TTBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TTBTBTTTTTTTTTTT"],
    "wide": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "narrow": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
}
RECONF_FORM_DC = {
    "s16x16": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s16x32": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s8x32": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s4x32": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "s1x32": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "bfp8": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "bfp4": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "bfpa": ["TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB", "TBBBBBBBBBBBBBBB"],
    "wide": ["BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB"],
    "narrow": ["BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB"],
}
NEXT_FORM_DC = {
    "s16x16": ["TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBBWBBBBBBBB", "TWWBBBBBBBBBBBBB"],
    "s16x32": ["TWWWBWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBBBBBBWWWWW", "TWWBBBBBBBBBBBBB"],
    "s8x32": ["TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBWWBBBBBBBB", "TWWBBBBBBBBBBBBB"],
    "s4x32": ["TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBWWBBBBBBBB", "TWWBBBBBBBBBBBBB"],
    "s1x32": ["TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWBBBWWBBBBBBBB", "TWWBBBBWWWWWWWWW"],
    "bfp8": ["TWBBBBWWWWWWWWWW", "TWBBBBBBBBBBBBBB", "TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW"],
    "bfp4": ["TWBBWWWWWWWWWWWW", "TWBBBBBBBBBBBBBB", "TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW"],
    "bfpa": ["TWBBWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW"],
    "wide": ["BWWWBBBBBBBBBBBB", "TWWWWWWWWWWWWWWW", "BWWWWWWWWWWWWWWW", "WWWWWWWWWWWWWWWW"],
    "narrow": ["BBBWWWWWWWWWWWWW", "TWWWWWWWWWWWWWWW", "BBBWWWWWWWWWWWWW", "WWWWWWWWWWWWWWWW"],
}
# Matmul masks: bit column * 16 + n - 1 at n tiles. [family][group][class]: family 0 datacopy, 1 32 bit input; groups LoFi K not given, LoFi K 1, LoFi K above 1, HiFi2 to HiFi4 (TAKE_W without the first group).
MM_NO_BLOCK = [[[9007336693694496, 524504, 0, 32, 0, 524456, 524328, 232, 128, 160], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [9007336693694496, 524504, 0, 32, 0, 524456, 524328, 232, 128, 160], [917758, 655530, 655594, 917710, 917742, 131170, 131106, 34, 64, 0]], [[9007336693694496, 786648, 0, 262176, 524288, 917672, 524328, 232, 128, 160], [0, 0, 0, 0, 0, 524288, 0, 0, 0, 0], [9007336693694496, 786648, 0, 262176, 524288, 917672, 524328, 232, 128, 160], [917758, 917674, 917738, 917710, 917742, 131170, 131106, 34, 64, 0]]]
MM_NO_W = [[[12948046497185824, 524504, 0, 96, 562949953421440, 42950197432, 562992903618600, 8589934824, 128, 160], [562958543355904, 0, 0, 0, 562949953421312, 0, 0, 0, 0, 0], [12948046497185824, 524504, 0, 96, 562949953421440, 42950197432, 562992903618600, 8589934824, 128, 160], [562949954339070, 917678, 655594, 917710, 917742, 131170, 131106, 34, 64, 131074]], [[12948046497185824, 786648, 262144, 262240, 562949953945728, 42950590648, 562992903618600, 8589934824, 655488, 131232], [562958543355904, 0, 0, 0, 562949953421312, 524288, 524288, 0, 0, 0], [12948046497185824, 786648, 262144, 262240, 562949953945728, 42950590648, 562992903618600, 8589934824, 655488, 131232], [562949954339070, 917678, 917738, 917710, 562949954339054, 917602, 131106, 34, 64, 131074]]]
MM_TAKE_W = [[[70086685085401088, 69814555957526544, 70086410207494144, 70086410207494144, 34057613188530176, 786492, 786444, 12, 240, 25769803782], [228698418577408, 69814555957526528, 70086410207494144, 70086410207494144, 34057613188530176, 262148, 0, 4, 112, 25769803776], [70086685085401088, 69814555957526528, 70086410207494144, 70086410207494144, 34057613188530176, 786460, 0, 0, 176, 25769803780]], [[70086685085401088, 69814555957526544, 70086410207494144, 70086410207494144, 34057613188530176, 60, 562949953814540, 12, 655600, 25770721286], [54271893947023360, 69814555957526528, 70086410207494144, 70086410207494144, 34057613188530176, 4, 393216, 4, 112, 25770590208], [70086685085401088, 69814555957526528, 70086410207494144, 70086410207494144, 34057613188530176, 28, 562949953421312, 0, 655536, 25770590212]]]
MM_TAKE_B = [[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0, 0, 0, 0, 524288], [0, 0, 0, 0, 0, 0, 0, 0, 0, 524288], [0, 0, 0, 0, 0, 524288, 0, 0, 0, 524288]]]
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


def pack_block_form(ntiles, dst_format, state, fp32_dest=False, full_sync=False, hifi=False, num_faces=4, face_r_dim=16, matmul_kt=0, src32=False):
    """Form of one llk_pack_block call; state is init (first block after a pack init or another output), reconf (first block after a packer reconfigure) or next (the block continues the output).
    src32: a Float32, Int32 or UInt32 CB sits in the kernel's CB list (the output's own CB counts, for its unpack source format is its format)."""
    if not fp32_dest and dst_format in ("Int8", "UInt8"):
        return FORM_NAMES["T"]
    family = 1 if (fp32_dest and (src32 or dst_format in ("Float32", "Int32", "UInt32"))) else 0
    table = {"init": (INIT_FORM_DC, INIT_FORM), "reconf": (RECONF_FORM_DC, RECONF_FORM), "next": (NEXT_FORM_DC, NEXT_FORM)}[state][family]
    column = (2 if full_sync else 0) + (1 if fp32_dest else 0)
    cls = output_class(dst_format, num_faces, face_r_dim)
    row = CLASSES.index(cls)
    count = min(ntiles, 16) - 1
    bit = column * 16 + count
    form = table[cls][column][count]
    group = 3 if hifi else (0 if matmul_kt == 0 else (1 if matmul_kt == 1 else 2))
    if form != "T" and (MM_NO_BLOCK[family][group][row] >> bit) & 1:
        form = "T"
    elif form == "W" and (MM_NO_W[family][group][row] >> bit) & 1:
        form = "B"
    if matmul_kt != 0 and state == "next" and form != "W" and (MM_TAKE_W[family][group - 1][row] >> bit) & 1:
        form = "W"
    if matmul_kt != 0 and state == "next" and form == "T" and (MM_TAKE_B[family][group - 1][row] >> bit) & 1:
        form = "B"
    return FORM_NAMES[form]


def test_mirror_cells():
    assert pack_block_form(1, 'Float16_b', 'init') == 'per_tile'
    assert pack_block_form(2, 'Float16_b', 'init') == 'block'
    assert pack_block_form(2, 'Float16_b', 'next') == 'block'
    assert pack_block_form(4, 'Float16_b', 'next', fp32_dest=True) == 'w_offset'
    assert pack_block_form(2, 'Bfp8_b', 'reconf') == 'block'
    assert pack_block_form(3, 'Bfp8_b', 'init') == 'block'
    assert pack_block_form(8, 'Bfp8_b', 'next') == 'per_tile'
    assert pack_block_form(12, 'Bfp8_b', 'next', full_sync=True) == 'w_offset'
    assert pack_block_form(16, 'Bfp8_b', 'init', full_sync=True) == 'block'
    assert pack_block_form(12, 'Float32', 'next', full_sync=True) == 'w_offset'
    assert pack_block_form(16, 'Float16_b', 'reconf', full_sync=True) == 'block'
    assert pack_block_form(4, 'Int8', 'init') == 'per_tile'
    assert pack_block_form(4, 'Int8', 'init', fp32_dest=True) == 'block'
    assert pack_block_form(3, 'UInt8', 'next') == 'per_tile'
    assert pack_block_form(16, 'Fp8_e4m3', 'next', full_sync=True) == 'w_offset'
    assert pack_block_form(3, 'Tf32', 'next') == 'w_offset'
    assert pack_block_form(4, 'Bfp8_b', 'next', matmul_kt=4) == 'per_tile'
    assert pack_block_form(4, 'Bfp8_b', 'next', matmul_kt=1) == 'w_offset'
    assert pack_block_form(4, 'Bfp8_b', 'next', hifi=True) == 'block'
    assert pack_block_form(4, 'Bfp8_b', 'next') == 'per_tile'
    assert pack_block_form(3, 'Float16_b', 'init', num_faces=2, face_r_dim=16) == 'block'
    assert pack_block_form(5, 'Float16_b', 'next', num_faces=2, face_r_dim=8) == 'w_offset'
    assert pack_block_form(3, 'Bfp8_b', 'next', fp32_dest=True) == 'block'
    assert pack_block_form(4, 'Bfp8_b', 'next', fp32_dest=True) == 'per_tile'
    assert pack_block_form(3, 'Bfp8_b', 'next', fp32_dest=True, src32=True) == 'per_tile'
    assert pack_block_form(4, 'Bfp8_b', 'next', fp32_dest=True, src32=True) == 'per_tile'
    assert pack_block_form(5, 'Float16_b', 'next', hifi=True, matmul_kt=0) == 'w_offset'
    assert pack_block_form(6, 'Bfp8_b', 'next', hifi=True, matmul_kt=0) == 'per_tile'
    assert pack_block_form(5, 'Float16_b', 'next', hifi=True, matmul_kt=1) == 'w_offset'
    assert pack_block_form(6, 'Bfp8_b', 'next', hifi=True, matmul_kt=1) == 'per_tile'
    assert pack_block_form(5, 'Float16_b', 'next', hifi=True, matmul_kt=4) == 'w_offset'
    assert pack_block_form(6, 'Bfp8_b', 'next', hifi=True, matmul_kt=4) == 'per_tile'


def _table(src, name):
    m = re.search(rf"constexpr char {name}\[10\]\[4\]\[17\] = \{{(.*?)\n\}};", src, re.S)
    assert m, name
    rows = re.findall(r"\{((?:\"[TBW]{16}\",? ?){4})\},\s*//\s*(.*)", m.group(1))
    return {CLASSES[i]: re.findall(r"\"([TBW]{16})\"", r) for i, (r, _) in enumerate(rows)}


def _masks(src, name):
    m = re.search(rf"constexpr std::uint64_t {name}\[2\]\[\d\]\[10\] = \{{(.*?)\n\}};", src, re.S)
    assert m, name
    fams = []
    for fb in re.findall(r"\{  //[^\n]*\n(.*?)\n    \},", m.group(1), re.S):
        fams.append([[int(x, 16) if x.startswith("0x") else int(x) for x in re.findall(r"0x[0-9A-F]{16}|\b0\b", g)] for g in re.findall(r"\{([^{}]*)\},", fb)])
    return fams


def test_header_tables_match_mirror():
    src = API.read_text()
    assert _table(src, "llk_pack_init_form") == INIT_FORM
    assert _table(src, "llk_pack_reconf_form") == RECONF_FORM
    assert _table(src, "llk_pack_next_form") == NEXT_FORM
    assert _table(src, "llk_pack_init_form_dc") == INIT_FORM_DC
    assert _table(src, "llk_pack_reconf_form_dc") == RECONF_FORM_DC
    assert _table(src, "llk_pack_next_form_dc") == NEXT_FORM_DC
    assert _masks(src, "llk_pack_mm_no_block") == MM_NO_BLOCK
    assert _masks(src, "llk_pack_mm_no_w") == MM_NO_W
    assert _masks(src, "llk_pack_mm_take_w") == MM_TAKE_W
    assert _masks(src, "llk_pack_mm_take_b") == MM_TAKE_B


def test_header_logic_matches_mirror():
    src = " ".join(API.read_text().split())
    assert "const std::uint32_t column = (DST_SYNC_MODE == DstSync::SyncFull ? 2 : 0) + (is_fp32_dest_acc_en ? 1 : 0);" in src
    assert "const std::uint32_t count = (ntiles < 16 ? ntiles : 16) - 1;" in src
    assert "if (!is_fp32_dest_acc_en && llk_pack_dst_is_int8(output_id)) { return 'T'; }" in src
    assert "const std::uint32_t bit = column * 16 + count;" in src
    assert "constexpr std::uint32_t family = llk_pack_src32_family<is_fp32_dest_acc_en> ? 1 : 0;" in src
    assert "inline constexpr bool llk_pack_src32_family = is_fp32_dest_acc_en && (llk_pack_src32_mask(0) != 0 || llk_pack_src32_mask(32) != 0);" in src
    assert "constexpr std::uint32_t group = MATH_FIDELITY != MathFidelity::LoFi ? 3 : (matmul_kt == 0 ? 0 : (matmul_kt == 1 ? 1 : 2));" in src
    assert "if (out != 'T' && ((llk_pack_mm_no_block[family][group][row] >> bit) & 1) != 0) { out = 'T'; } else if (out == 'W' && ((llk_pack_mm_no_w[family][group][row] >> bit) & 1) != 0) { out = 'B'; }" in src
    assert "if (next && out != 'W' && ((llk_pack_mm_take_w[family][group - 1][row] >> bit) & 1) != 0) { out = 'W'; }" in src
    assert "if (next && out == 'T' && ((llk_pack_mm_take_b[family][group - 1][row] >> bit) & 1) != 0) { out = 'B'; }" in src
    assert "if (get_output_num_faces(output_id) != 4 || rows != FACE_R_DIM) { if (rows == FACE_R_DIM) { return get_output_num_faces(output_id) == 1 ? 0 : 1; } return rows >= 8 ? 2 : (rows >= 4 ? 3 : 4); }" in src
    assert "return llk_pack_dst_is_wide(output_id) ? 8 : 9;" in src
    assert "first_block ? (pack_reconfigured ? llk_pack_form_table<is_fp32_dest_acc_en>(llk_pack_reconf_form, llk_pack_reconf_form_dc)" in src
    assert "closed_pays" not in src and "closed_loses" not in src and "LLK_PACK_BLOCK_MIN_TILES" not in src


def _alt_letter(family, cls, col, count):
    letter = RECONF_FORM[cls][col][count] if family else RECONF_FORM_DC[cls][col][count]
    if letter != "T" and (MM_NO_BLOCK[family][0][CLASSES.index(cls)] >> (col * 16 + count)) & 1:
        return "T"
    return letter


def test_alt_kernel_mirrors_reconf_rows():
    """pack_block_alt_test.cpp holds the reconfigure rows of its three formats (Bfp8_b, Float32, Float16_b) as llk_pack_block takes them with the matmul K not given, for a kernel with and without a 32 bit CB."""
    src = ALT.read_text()
    m = re.search(r"constexpr char alt_reconf_form\[2\]\[3\]\[4\]\[17\] = \{(.*?)\n\};", src, re.S)
    assert m
    rows = [re.findall(r"\"([TBW]{16})\"", r) for r in re.findall(r"\{((?:\"[TBW]{16}\",? ?){4})\}", m.group(1))]
    expect = [["".join(_alt_letter(f, c, col, n) for n in range(16)) for col in range(4)] for f in (0, 1) for c in ("bfp8", "wide", "narrow")]
    assert rows == expect
