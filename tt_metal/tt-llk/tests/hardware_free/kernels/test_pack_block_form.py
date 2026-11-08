# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Python mirror of the form llk_pack_block takes per cell (tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_tile_api.h), checked against the header.

A block takes the per-tile pack (main's code, T), the plain or closed block that programs the output (B) or the block with the W offset (W). The form tables hold one row per output
class (small tiles, Bfp8_b Bfp4_b Bfp2_b Bfp8, Bfp4 and Bfp2, 4 byte datums, 2 byte datums, other formats) and one column per DEST width and sync, from the tt-llk crossover sweeps. The first block of
an output takes the init or the reconfigure table, the blocks that continue it the next table; block-float outputs next to HiFi math at half sync stay per tile.
"""

import pathlib
import re

API = pathlib.Path(__file__).resolve().parents[4] / "hw/ckernels/blackhole/metal/llk_api/llk_pack_tile_api.h"
ALT = pathlib.Path(__file__).resolve().parents[2] / "sources/pack_block_alt_test.cpp"
CLASSES = ("small", "bfp", "bfpa", "wide", "narrow", "other")

INIT_FORM = {
    "small": ["TTBBBBBB", "TTBBBBBB", "TBBBBBBB", "TBBBBBBB"],
    "bfp": ["TTBBBBBB", "TTBBBBBB", "TTBBBBBB", "TTBBBBBB"],
    "bfpa": ["TTBBBBBB", "TTBBBBBB", "TBBBBBBB", "TTBTBTTT"],
    "wide": ["TBBBBBBB", "TBBBBBBB", "BBBBBBBB", "TBBBBBBB"],
    "narrow": ["TBBBBBBB", "TBBBBBBB", "BBBBBBBB", "TBBBBBBB"],
    "other": ["TTTBBBBB", "TTTBBBBB", "TTTBBBBB", "TTTBBBBB"],
}
RECONF_FORM = {
    "small": ["TBBBBBBB", "TBBBBBBB", "TBBBBBBB", "TBBBBBBB"],
    "bfp": ["TBBBBBBB", "TBTTTTTT", "TBBBBBBB", "TTBBBBBB"],
    "bfpa": ["TBBBBBBB", "TBBBBBBB", "TBBBBBBB", "TBBBBBBB"],
    "wide": ["BBBBBBBB", "BBTTTTTT", "BBBBBBBB", "BBBBBBBB"],
    "narrow": ["BBBBBBBB", "BBBBBBBB", "BBBBBBBB", "BBBBBBBB"],
    "other": ["TTTBBBBB", "TTTBBBBB", "TTTBBBBB", "TTTBBBBB"],
}
NEXT_FORM = {
    "small": ["TWWWBWWW", "TWWWWWWW", "TWWBBBBB", "TWWBBBBB"],
    "bfp": ["TWBBBBWW", "TBWTTTTT", "TWWWWWWW", "TWWWWWWW"],
    "bfpa": ["TWBBWWWW", "TWWWWWWW", "TWWWWWWW", "TWWWWWWW"],
    "wide": ["WWWWBBBB", "TBWBBBBB", "WWWWWWWW", "WWWWWWWW"],
    "narrow": ["WWWWWWWW", "TBBBBBBB", "WWWWWWWW", "WWWWWWWW"],
    "other": ["TTTWWWWW", "TTTWWWWW", "TTTWWWWW", "TTTWWWWW"],
}
FORM_NAMES = {"T": "per_tile", "B": "block", "W": "w_offset"}
BFP_B = ("Bfp8_b", "Bfp4_b", "Bfp2_b", "Bfp8")
WIDE = ("Float32", "Int32", "UInt32", "Tf32")
NARROW = ("Float16_b", "Float16", "UInt16")


def output_class(dst_format, num_faces=4, face_r_dim=16):
    if num_faces != 4 or face_r_dim != 16:
        return "small"
    if dst_format in ("Bfp4", "Bfp2"):
        return "bfpa"
    if dst_format in BFP_B:
        return "bfp"
    if dst_format in WIDE:
        return "wide"
    return "narrow" if dst_format in NARROW else "other"


def pack_block_form(ntiles, dst_format, state, fp32_dest=False, full_sync=False, hifi=False, num_faces=4, face_r_dim=16):
    """Form of one llk_pack_block call; state is init (first block after a pack init or another output), reconf (first block after a packer reconfigure) or next (the block continues the output)."""
    if hifi and not full_sync and dst_format in BFP_B + ("Bfp4", "Bfp2"):
        return "per_tile"
    table = {"init": INIT_FORM, "reconf": RECONF_FORM, "next": NEXT_FORM}[state]
    column = (2 if full_sync else 0) + (1 if fp32_dest else 0)
    return FORM_NAMES[table[output_class(dst_format, num_faces, face_r_dim)][column][min(ntiles, 8) - 1]]


def test_mirror_cells():
    assert pack_block_form(1, "Float16_b", "init") == "per_tile"
    assert pack_block_form(2, "Float16_b", "init") == "block"
    assert pack_block_form(2, "Float16_b", "next") == "w_offset"
    assert pack_block_form(4, "Float16_b", "next", fp32_dest=True) == "block"
    assert pack_block_form(4, "Float16_b", "next", fp32_dest=True, full_sync=True) == "w_offset"
    assert pack_block_form(2, "Bfp8_b", "init") == "per_tile"
    assert pack_block_form(2, "Bfp8_b", "reconf") == "block"
    assert pack_block_form(3, "Bfp8_b", "init") == "block"
    assert pack_block_form(2, "Bfp8_b", "next") == "w_offset"
    assert pack_block_form(3, "Bfp8_b", "next") == "block"
    assert pack_block_form(3, "Bfp8_b", "init", fp32_dest=True) == "block"
    assert pack_block_form(3, "Bfp8_b", "reconf", fp32_dest=True) == "per_tile"
    assert pack_block_form(4, "Bfp8_b", "next", fp32_dest=True) == "per_tile"
    assert pack_block_form(4, "Bfp8_b", "next", fp32_dest=True, full_sync=True) == "w_offset"
    assert pack_block_form(5, "Bfp8_b", "next") == "block"
    assert pack_block_form(4, "Bfp8_b", "next", hifi=True) == "per_tile"
    assert pack_block_form(4, "Bfp8_b", "next", hifi=True, full_sync=True) == "w_offset"
    assert pack_block_form(4, "Float32", "init", fp32_dest=True) == "block"
    assert pack_block_form(1, "Float32", "init", fp32_dest=True) == "per_tile"
    assert pack_block_form(1, "Float32", "reconf", fp32_dest=True) == "block"
    assert pack_block_form(2, "Float16_b", "init", num_faces=2) == "per_tile"
    assert pack_block_form(3, "Float16_b", "init", num_faces=2) == "block"
    assert pack_block_form(3, "Int8", "init") == "per_tile"
    assert pack_block_form(4, "Int8", "init") == "block"
    assert pack_block_form(12, "Bfp8_b", "next", full_sync=True) == pack_block_form(8, "Bfp8_b", "next", full_sync=True)


def _table(src, name):
    m = re.search(rf"constexpr char {name}\[6\]\[4\]\[9\] = \{{(.*?)\n\}};", src, re.S)
    assert m, name
    rows = re.findall(r"\{((?:\"[TBW]{8}\",? ?){4})\},\s*//\s*(\w+)", m.group(1))
    return {k: re.findall(r"\"([TBW]{8})\"", r) for r, k in rows}


def test_header_tables_match_mirror():
    src = API.read_text()
    assert _table(src, "llk_pack_init_form") == INIT_FORM
    assert _table(src, "llk_pack_reconf_form") == RECONF_FORM
    assert _table(src, "llk_pack_next_form") == NEXT_FORM


def test_header_logic_matches_mirror():
    src = " ".join(API.read_text().split())
    assert "if (MATH_FIDELITY != MathFidelity::LoFi && DST_SYNC_MODE == DstSync::SyncHalf && llk_pack_dst_is_bfp(output_id)) { return 'T'; }" in src
    assert "const std::uint32_t column = (DST_SYNC_MODE == DstSync::SyncFull ? 2 : 0) + (is_fp32_dest_acc_en ? 1 : 0);" in src
    assert "table[llk_pack_block_class(output_id)][column][(ntiles < 8 ? ntiles : 8) - 1]" in src
    assert "if (get_output_num_faces(output_id) != 4 || get_output_face_r_dim(output_id) != FACE_R_DIM) { return 0; }" in src
    assert "return llk_pack_dst_is_wide(output_id) ? 3 : (llk_pack_dst_is_narrow(output_id) ? 4 : 5);" in src
    assert "first_block ? (pack_reconfigured ? llk_pack_reconf_form : llk_pack_init_form) : llk_pack_next_form" in src
    assert "closed_pays" not in src and "closed_loses" not in src and "LLK_PACK_BLOCK_MIN_TILES" not in src


def test_alt_kernel_mirrors_reconf_rows():
    """pack_block_alt_test.cpp holds the reconfigure rows of its three formats (Bfp8_b, Float32, Float16_b)."""
    src = ALT.read_text()
    m = re.search(r"constexpr char alt_reconf_form\[3\]\[4\]\[9\] = \{(.*?)\n\};", src, re.S)
    rows = [re.findall(r"\"([TBW]{8})\"", r) for r in re.findall(r"\{((?:\"[TBW]{8}\",? ?){4})\}", m.group(1))]
    assert rows == [RECONF_FORM["bfp"], RECONF_FORM["wide"], RECONF_FORM["narrow"]]
