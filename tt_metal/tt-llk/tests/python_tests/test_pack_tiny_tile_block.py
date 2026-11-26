# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test: Pack tiny tiles in contiguous L1 blocks.

Validates packing multi-tile blocks of tiny tiles (1x32, 8x32, 16x32,
16x16) as well as standard 32x32 tiles into contiguous L1 memory.

In the kernel used by this test, math writes each tile into sparse
Tile32x32 DEST slots, and _llk_pack_block_contiguous_ performs the
sparse-to-dense packing into a contiguous L1 block.
"""

from conftest import skip_for_wormhole
from helpers.llk_params import DestAccumulation
from helpers.pack_tiny_tile import FORMATS, TILE_DIMS, make_pack_tiny_tile_config
from helpers.param_config import parametrize
from helpers.utils import passed_test, result_tensor

PACK_TINY_TILE_BLOCK_SWEEP = dict(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    tile_dims=TILE_DIMS,
    num_tiles=[1, 2, 4, 8],
)


@skip_for_wormhole
@parametrize(**PACK_TINY_TILE_BLOCK_SWEEP)
def test_pack_tiny_tile_block(
    formats,
    dest_acc,
    tile_dims,
    num_tiles,
    *,
    is_perf=False,
    perf_report=None,
    run_types=None,
    loop_factor=1,
):
    outcome, golden_tensor = make_pack_tiny_tile_config(
        tile_dims,
        num_tiles,
        formats,
        dest_acc,
        test_name="sources/pack_tiny_tile_block_test.cpp",
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
    if is_perf:
        return

    res_tensor = result_tensor(outcome.result, formats.output_format, golden_tensor)
    assert passed_test(golden_tensor, res_tensor, formats.output_format)
