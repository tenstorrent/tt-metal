# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf sweep of the tiny-tile and partial-face packs (pack_tiny_tile_block_test.cpp).

perf_pack and perf_pack_cov2 time 32x32 tiles only. This module times the pack of blocks of 16x32, 8x32, 4x32, 1x32 (two faces of 16, 8, 4 and 1 rows, the
last four are partial-face tiles) and the one-face 16x16 tile, two, four and eight tiles per run, with
  * the per-tile _llk_pack_ (mode 2),
  * the closed block _llk_pack_block_closed_, one close per tile (mode 1),
  * the dense replay block _llk_pack_block_dense_, one close per run (mode 3),
  * the experimental contiguous block (mode 0) for plain formats.
Float16_b and Float32 (16-bit and 32-bit DEST), and Bfp8_b output for the 16-row faces, the only block-float tiles the packer writes correctly."""

import pytest
from conftest import skip_for_wormhole
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import DestAccumulation
from helpers.param_config import parametrize
from helpers.perf.core import PerfRunType
from test_pack_tiny_tile_block import _make_config

F = DataFormat
TINY_PERF_FORMATS = [
    InputOutputFormat(F.Float16_b, F.Float16_b),
    InputOutputFormat(F.Float32, F.Float32),
    InputOutputFormat(F.Float16_b, F.Bfp8_b),
]
TINY_PERF_DIMS = [(16, 32), (8, 32), (4, 32), (1, 32), (16, 16)]
TINY_PERF_RUN_TYPES = [PerfRunType.PACK_ISOLATE, PerfRunType.L1_TO_L1]


@pytest.mark.perf
@skip_for_wormhole
@parametrize(
    formats=TINY_PERF_FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    tile_dims=TINY_PERF_DIMS,
    num_tiles=[2, 4, 8],
    pack_mode=[0, 1, 2, 3, 4],
    loop_factor=[32],
)
def test_perf_pack_tiny_tile_block(perf_report, formats, dest_acc, tile_dims, num_tiles, pack_mode, loop_factor):
    if formats.output_format.is_block_float() and (tile_dims[0] != 16 or tile_dims[1] != 32 or pack_mode in (0, 3)):
        pytest.skip("Block-float tiles need two 16-row faces and keep the tile close")
    configuration, _, _ = _make_config(
        tile_dims,
        num_tiles,
        formats,
        dest_acc,
        pack_mode,
        loop_factor=loop_factor,
        is_perf=True,
        run_types=TINY_PERF_RUN_TYPES,
    )
    configuration.run(perf_report)
