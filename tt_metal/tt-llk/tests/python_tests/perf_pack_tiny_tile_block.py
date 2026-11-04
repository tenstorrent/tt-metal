# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_wormhole
from helpers.llk_params import DestAccumulation
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_pack_tiny_tile_block import (
    FORMATS,
)
from test_pack_tiny_tile_block import (
    test_pack_tiny_tile_block as _test_pack_tiny_tile_block,
)
from test_pack_tiny_tile_block import (
    test_pack_tiny_tile_reconfig as _test_pack_tiny_tile_reconfig,
)


@pytest.mark.perf
@skip_for_wormhole
@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    tile_dims=[
        (1, 32),
        (2, 32),
        (4, 32),
        (8, 32),
        (16, 32),
        (16, 16),
        (32, 32),
    ],
    num_tiles=[1, 2, 4, 8],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_pack_tiny_tile_block(
    perf_report,
    formats,
    dest_acc,
    tile_dims,
    num_tiles,
    run_types,
    loop_factor,
    is_perf,
):
    _test_pack_tiny_tile_block(
        formats,
        dest_acc,
        tile_dims,
        num_tiles,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )


@pytest.mark.perf
@skip_for_wormhole
@parametrize(
    formats=FORMATS,
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    tile_dims=[
        (1, 32),
        (2, 32),
        (4, 32),
        (8, 32),
        (16, 32),
        (16, 16),
    ],
    num_tiles=[2, 4, 8],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_pack_tiny_tile_reconfig(
    perf_report,
    formats,
    dest_acc,
    tile_dims,
    num_tiles,
    run_types,
    loop_factor,
    is_perf,
):
    _test_pack_tiny_tile_reconfig(
        formats,
        dest_acc,
        tile_dims,
        num_tiles,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
