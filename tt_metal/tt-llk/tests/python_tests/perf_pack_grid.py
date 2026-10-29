# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The block-size and integer-output part of the Blackhole pack API space that perf_pack, perf_pack_cov2 and perf_pack_cov3 leave out.

  * block sizes of 3, 5, 6, 7 and 9 to 15 tiles (cov3 holds 1, 2, 4, 8 and 16) and the default block at 4, 8 and 16 tiles from dest index 0: the per-tile pack of the
    same key as the twin, the llk_pack_block default calls (pack_block 2: blocks under four tiles take the per-tile pack, block-float blocks the closed form) and
    the W offset calls (pack_block 3), from dest index 0, 1, 3 and the last tile that fits;
  * the integer outputs UInt16, UInt32, UInt8 and Int8 that cov3 does not run, same-format, at the same sizes and forms.
The block size is in num_tiles_in_block and no key repeats one of the other pack perf modules. perf_pack_grid_rows holds the row pack, perf_pack_grid_tiny the tiny tiles."""

import pytest
from conftest import skip_for_wormhole
from helpers.constraints import get_valid_dest_accumulation_modes
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation, DestSync, PackerReluType
from helpers.param_config import DEST_SYNC_TILE_LIMITS, effective_dest_acc, parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_pack import test_pack as run_pack

F = DataFormat
IO = InputOutputFormat
# Output format by DEST width: the pack path depends on the output format and on whether the packer reads 16 or 32 bit DEST, so each output format meets a 16-bit and a 32-bit input.
GRID_FLOAT_FORMATS = [
    IO(F.Float16_b, F.Float16_b),
    IO(F.Float16, F.Float16),
    IO(F.Float32, F.Float32),
    IO(F.Int32, F.Int32),
    IO(F.Bfp8_b, F.Bfp8_b),
    IO(F.Bfp4_b, F.Bfp4_b),
    IO(F.Float16_b, F.Bfp8_b),
    IO(F.Float32, F.Bfp8_b),
    IO(F.Float16_b, F.Bfp4_b),
    IO(F.Float32, F.Bfp4_b),
    IO(F.Float16_b, F.Float32),
    IO(F.Float32, F.Float16_b),
    IO(F.Float16_b, F.Float16),
    IO(F.Float32, F.Float16),
]
GRID_INT_FORMATS = [IO(F.UInt16, F.UInt16), IO(F.UInt32, F.UInt32), IO(F.UInt8, F.UInt8), IO(F.Int8, F.Int8)]
GRID_NEW_BLOCKS = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
GRID_DEFAULT_BLOCKS = [4, 8, 16]
GRID_BLOCKS = sorted(GRID_NEW_BLOCKS + GRID_DEFAULT_BLOCKS)


def _section_tiles(dest_sync, dest_acc, formats):
    wide = effective_dest_acc(formats.input_format, formats.output_format, dest_acc) == DestAccumulation.Yes or formats.input_format.is_32_bit()
    return DEST_SYNC_TILE_LIMITS[dest_sync] // (2 if wide else 1)


def _grid_indices(dest_acc, dest_sync, formats, block_tiles):
    section = _section_tiles(dest_sync, dest_acc, formats)
    if block_tiles > section:
        return []
    return sorted({i for i in (0, 1, 3, section - block_tiles) if i + block_tiles <= section})


def _grid_dims(block_tiles):
    return [[64, 32 * block_tiles]]


def _grid_kinds(block_tiles):
    # blocks of fewer than four tiles take the per-tile pack in the default block call, and in the W offset call: one block row of the pair is enough
    return [False, 2] if block_tiles < 4 else [False, 2, 3]


def _grid_blocks(dest_acc, dest_sync, formats):
    section = _section_tiles(dest_sync, dest_acc, formats)
    return [n for n in GRID_BLOCKS if n <= section]


GRID_SWEEP = dict(
    formats=GRID_FLOAT_FORMATS + GRID_INT_FORMATS,
    dest_acc=get_valid_dest_accumulation_modes,
    relu_type=[PackerReluType.NoRelu],
    dest_sync=[DestSync.Half, DestSync.Full],
    block_tiles=_grid_blocks,
    input_dimensions=_grid_dims,
    dest_index=_grid_indices,
    pack_block=_grid_kinds,
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
    is_perf=[True],
)


@pytest.mark.perf
@skip_for_wormhole
@parametrize(**GRID_SWEEP)
def test_perf_pack_grid_block(
    perf_report, formats, dest_acc, relu_type, dest_sync, block_tiles, input_dimensions, dest_index, pack_block, run_types, loop_factor, is_perf
):
    """Per-tile, default block and W offset packs of block_tiles tiles from dest_index."""
    run_pack(
        formats,
        dest_acc,
        input_dimensions,
        relu_type,
        dest_sync,
        dest_index,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
        pack_block=pack_block,
        block_tiles=block_tiles,
    )
