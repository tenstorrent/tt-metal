# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Python mirror of the form llk_pack_block takes per cell (tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_tile_api.h), checked against the header.

A block takes the per-tile pack (main's code), the plain block, or the W offset block. Below LLK_PACK_W_MIN_TILES the W setup is not paid back and the plain block is used;
Float32 outputs of 4 tiles always, and of 5 tiles for the first block of an output, take the per-tile pack (the plain block is slower there).
"""

import pathlib
import re

API = pathlib.Path(__file__).resolve().parents[4] / "hw/ckernels/blackhole/metal/llk_api/llk_pack_tile_api.h"
W_MIN_TILES = 5
BLOCK_MIN_TILES = 4


def pack_block_form(ntiles, dst_format, first_block_of_output, small_tile=False, contiguous=True):
    """Form of one llk_pack_block call whose block form is taken (contiguous, plain format, full tiles)."""
    if ntiles < BLOCK_MIN_TILES and not (ntiles >= 2 and small_tile):
        return "per_tile"
    if dst_format == "Float32" and ntiles <= 5 and (ntiles == 4 or first_block_of_output):
        return "per_tile"
    if ntiles < W_MIN_TILES or first_block_of_output:
        return "plain"
    return "w_offset"


def test_mirror_cells():
    assert pack_block_form(3, "Float16_b", False) == "per_tile"
    assert pack_block_form(4, "Float16_b", False) == "plain"
    assert pack_block_form(4, "Float32", False) == "per_tile"
    assert pack_block_form(5, "Float32", True) == "per_tile"
    assert pack_block_form(5, "Float32", False) == "w_offset"
    assert pack_block_form(8, "Float16_b", True) == "plain"
    assert pack_block_form(8, "Float16_b", False) == "w_offset"


def test_header_matches_mirror():
    src = API.read_text()
    assert re.search(rf"#define LLK_PACK_W_MIN_TILES {W_MIN_TILES}\b", src)
    assert re.search(rf"#define LLK_PACK_BLOCK_MIN_TILES {BLOCK_MIN_TILES}\b", src)
    assert "ntiles <= 5 && static_cast<DataFormat>(pack_dst_format[output_id]) == DataFormat::Float32 && (ntiles == 4 || pack_w_last != output_id)" in src
    assert "if (ntiles < LLK_PACK_W_MIN_TILES) {" in src
