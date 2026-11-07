# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Python mirror of the form llk_pack_block takes per cell (tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_pack_tile_api.h), checked against the header.

A block takes the per-tile pack (main's code), the plain block, or the W offset block. The first block of an output (right after a pack init, or after another output) takes the per-tile pack
for 32 bit outputs (Float32, Int32, UInt32, Tf32) up to 8 tiles, for a 4 tile block-float block and for two small tiles, where the plain block is slower; every other first block is the plain block and the next ones take the W offset.
Four Bfp8_b tiles from an fp32 DEST at half sync take the closed block without W on every call: the W block's back to back writes take L1 bank cycles from the unpacker's Float32 reads.
"""

import pathlib
import re

API = (
    pathlib.Path(__file__).resolve().parents[4]
    / "hw/ckernels/blackhole/metal/llk_api/llk_pack_tile_api.h"
)
BLOCK_MIN_TILES = 4
WIDE_FORMATS = ("Float32", "Int32", "UInt32", "Tf32")  # datum_size_in_bytes == 4


def pack_block_form(
    ntiles,
    dst_format,
    first_block_of_output,
    small_tile=False,
    bfp=False,
    fp32_dest=False,
    half_sync=False,
):
    """Form of one llk_pack_block call of a plain-format or block-float output with full tiles."""
    if ntiles < BLOCK_MIN_TILES and not (ntiles >= 2 and small_tile):
        return "per_tile"
    if first_block_of_output and (
        (dst_format in WIDE_FORMATS and ntiles <= 8)
        or (bfp and ntiles == 4)
        or (ntiles == 2 and small_tile)
    ):
        return "per_tile"
    if fp32_dest and half_sync and ntiles == 4 and dst_format == "Bfp8_b":
        return "closed"
    return "plain" if first_block_of_output else "w_offset"


def test_mirror_cells():
    assert pack_block_form(3, "Float16_b", False) == "per_tile"
    assert pack_block_form(4, "Float16_b", True) == "plain"
    assert pack_block_form(4, "Float16_b", False) == "w_offset"
    assert pack_block_form(4, "Float32", True) == "per_tile"
    assert pack_block_form(4, "Float32", False) == "w_offset"
    assert pack_block_form(8, "Float32", True) == "per_tile"
    assert pack_block_form(8, "Tf32", True) == "per_tile"
    assert pack_block_form(4, "Int32", True) == "per_tile"
    assert pack_block_form(12, "Float32", True) == "plain"
    assert pack_block_form(4, "Bfp8_b", True, bfp=True) == "per_tile"
    assert pack_block_form(8, "Bfp8_b", True, bfp=True) == "plain"
    assert pack_block_form(2, "Float16_b", True, small_tile=True) == "per_tile"
    assert pack_block_form(2, "Float16_b", False, small_tile=True) == "w_offset"
    assert pack_block_form(3, "Float16_b", True, small_tile=True) == "plain"
    kw = dict(bfp=True, fp32_dest=True, half_sync=True)
    assert pack_block_form(4, "Bfp8_b", False, **kw) == "closed"
    assert pack_block_form(4, "Bfp4_b", False, **kw) == "w_offset"
    assert pack_block_form(4, "Bfp8_b", False, bfp=True, fp32_dest=True) == "w_offset"


def test_header_matches_mirror():
    src = " ".join(API.read_text().split())
    assert re.search(rf"#define LLK_PACK_BLOCK_MIN_TILES {BLOCK_MIN_TILES}\b", src)
    assert "LLK_PACK_W_MIN_TILES" not in src
    assert (
        "if (first_block && ((ntiles <= 8 && datum_size_in_bytes(pack_dst_format[output_id]) == 4) || (ntiles == 4 && llk_pack_dst_is_bfp(output_id)) || (ntiles == 2 && llk_pack_block_is_small(output_id)))) {"
        in src
    )
    assert (
        "if (is_fp32_dest_acc_en && DST_SYNC_MODE == DstSync::SyncHalf && ntiles == 4 && pack_dst_format[output_id] == static_cast<std::uint32_t>(DataFormat::Bfp8_b)) {"
        in src
    )
