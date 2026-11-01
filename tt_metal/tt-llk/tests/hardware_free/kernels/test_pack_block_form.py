# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Python mirror of the form llk_pack_block takes per cell (tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_tile_api.h), checked against the header.

A block takes the per-tile pack (main's code), the plain block, or the W offset block. The first block of an output (right after a pack init, or after another output) takes the per-tile pack
for Float32 up to 7 tiles and for a 4 tile block-float block, where the plain block is slower; every other first block is the plain block and the next ones take the W offset.
"""

import pathlib
import re

API = pathlib.Path(__file__).resolve().parents[4] / "hw/ckernels/blackhole/metal/llk_api/llk_pack_tile_api.h"
BLOCK_MIN_TILES = 4


def pack_block_form(ntiles, dst_format, first_block_of_output, small_tile=False, bfp=False):
    """Form of one llk_pack_block call of a plain-format or block-float output with full tiles."""
    if ntiles < BLOCK_MIN_TILES and not (ntiles >= 2 and small_tile):
        return "per_tile"
    if first_block_of_output and ((dst_format == "Float32" and ntiles <= 7) or (bfp and ntiles == 4)):
        return "per_tile"
    return "plain" if first_block_of_output else "w_offset"


def test_mirror_cells():
    assert pack_block_form(3, "Float16_b", False) == "per_tile"
    assert pack_block_form(4, "Float16_b", True) == "plain"
    assert pack_block_form(4, "Float16_b", False) == "w_offset"
    assert pack_block_form(4, "Float32", True) == "per_tile"
    assert pack_block_form(4, "Float32", False) == "w_offset"
    assert pack_block_form(7, "Float32", True) == "per_tile"
    assert pack_block_form(8, "Float32", True) == "plain"
    assert pack_block_form(4, "Bfp8_b", True, bfp=True) == "per_tile"
    assert pack_block_form(8, "Bfp8_b", True, bfp=True) == "plain"


def test_header_matches_mirror():
    src = API.read_text()
    assert re.search(rf"#define LLK_PACK_BLOCK_MIN_TILES {BLOCK_MIN_TILES}\b", src)
    assert "LLK_PACK_W_MIN_TILES" not in src
    assert "if (first_block && ((ntiles <= 7 && static_cast<DataFormat>(pack_dst_format[output_id]) == DataFormat::Float32) || (ntiles == 4 && llk_pack_dst_is_bfp(output_id)))) {" in src
