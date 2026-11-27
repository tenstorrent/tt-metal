# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Functional tests for packer reconfiguration from 32x32 to tiny tiles."""

from conftest import skip_for_wormhole
from helpers.llk_params import DestAccumulation
from helpers.pack_tiny_tile import FORMATS, TILE_DIMS, run_pack_tiny_tile
from helpers.param_config import parametrize
from helpers.utils import passed_test, result_tensor

RECONFIG_TILE_DIMS = TILE_DIMS[:-1]

PACK_TINY_TILE_RECONFIG_SWEEP = dict(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    tile_dims=RECONFIG_TILE_DIMS,
    num_tiles=[2, 4, 8],
)


@skip_for_wormhole
@parametrize(**PACK_TINY_TILE_RECONFIG_SWEEP)
def test_pack_tiny_tile_reconfig(
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
    """Reconfigure 32x32 pack state to tiny tiles, then pack a multi-tile block."""
    outcome, golden_tensor = run_pack_tiny_tile(
        tile_dims,
        num_tiles,
        formats,
        dest_acc,
        test_name="sources/pack_tiny_tile_reconfig_test.cpp",
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
    if is_perf:
        return

    res_tensor = result_tensor(outcome.result, formats.output_format, golden_tensor)
    assert passed_test(golden_tensor, res_tensor, formats.output_format)
