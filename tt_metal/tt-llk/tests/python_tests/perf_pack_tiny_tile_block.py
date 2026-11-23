# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_wormhole
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_pack_tiny_tile_block import (
    PACK_TINY_TILE_BLOCK_SWEEP,
)
from test_pack_tiny_tile_block import (
    test_pack_tiny_tile_block as _test_pack_tiny_tile_block,
)


@pytest.mark.perf
@skip_for_wormhole
@parametrize(**PACK_TINY_TILE_BLOCK_SWEEP, **PERF_MEASUREMENT_AXES)
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
