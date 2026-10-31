# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tiny tiles and partial faces that perf_pack_tiny_tile_block does not run: 2x32 and 32x16 tiles, block sizes of 1 to 8 tiles, the first or the last dest tile that
fits as the start, Float16 and UInt16 outputs, and Bfp4_b for the 16x32 tile; modes 1 (closed block, library form), 2 (per-tile), 3 (dense block) and 4 (the closed block llk_pack_block makes, with the W offset). Four tile blocks from index 0 of
the old shapes and formats are rows of perf_pack_tiny_tile_block and are not repeated."""

import pytest
from conftest import skip_for_wormhole
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation
from helpers.perf.core import ALL_PERF_RUN_TYPES
from helpers.param_config import parametrize
from test_pack_tiny_tile_block import _make_config as _run_tiny_config

F = DataFormat
IO = InputOutputFormat
TINY_GRID_DIMS = [(1, 32), (2, 32), (4, 32), (8, 32), (16, 32), (16, 16), (32, 16)]
TINY_GRID_FORMATS = [
    IO(F.Float16_b, F.Float16_b),
    IO(F.Float16, F.Float16),
    IO(F.Float32, F.Float32),
    IO(F.UInt16, F.UInt16),
    IO(F.Float16_b, F.Bfp8_b),
    IO(F.Float16_b, F.Bfp4_b),
]
TINY_GRID_BLOCKS = [1, 2, 3, 4, 5, 6, 7, 8]
TINY_OLD_DIMS = [(16, 32), (8, 32), (4, 32), (1, 32), (16, 16)]
TINY_OLD = {("Float16_b", "Float16_b"), ("Float32", "Float32"), ("Float16_b", "Bfp8_b")}


def _tiny_blocks(formats, dest_acc, tile_dims):
    wide = dest_acc == DestAccumulation.Yes or formats.input_format.is_32_bit() or formats.output_format.is_32_bit()
    return [n for n in TINY_GRID_BLOCKS if n <= (4 if wide else 8)]


def _tiny_indices(formats, dest_acc, tile_dims, block_tiles):
    wide = dest_acc == DestAccumulation.Yes or formats.input_format.is_32_bit() or formats.output_format.is_32_bit()
    section = 4 if wide else 8
    return sorted({i for i in (0, section - block_tiles) if i + block_tiles <= section})


def _tiny_modes(formats, tile_dims, block_tiles, dest_index):
    if formats.output_format.is_block_float():
        return [1, 2, 4] if tile_dims == (16, 32) else []
    if (formats.input_format.name, formats.output_format.name) in TINY_OLD and dest_index == 0 and block_tiles == 4 and tile_dims in TINY_OLD_DIMS:
        return []
    return [1, 2, 3, 4]


@pytest.mark.perf
@skip_for_wormhole
@parametrize(
    formats=TINY_GRID_FORMATS,
    dest_acc=get_valid_dest_accumulation_modes,
    tile_dims=TINY_GRID_DIMS,
    block_tiles=_tiny_blocks,
    dest_index=_tiny_indices,
    pack_mode=_tiny_modes,
    loop_factor=[32],
)
def test_perf_pack_grid_tiny(perf_report, formats, dest_acc, tile_dims, block_tiles, dest_index, pack_mode, loop_factor):
    """Tiny tiles and partial faces: per-tile (mode 2), closed block (mode 1) and dense block (mode 3) of block_tiles tiles from dest_index."""
    configuration, _, _ = _run_tiny_config(
        tile_dims,
        2 * block_tiles,
        formats,
        dest_acc,
        pack_mode,
        loop_factor=loop_factor,
        is_perf=True,
        run_types=ALL_PERF_RUN_TYPES,
        block_tiles=block_tiles,
        dest_index=dest_index,
    )
    configuration.run(perf_report)
