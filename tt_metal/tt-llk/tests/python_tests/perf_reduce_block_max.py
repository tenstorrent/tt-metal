# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_reduce_block_max import (
    FORMATS,
    _run_reduce_block_max,
    block_max_tile_dimensions,
)


@pytest.mark.perf
@parametrize(
    formats=FORMATS,
    block_ct_dim=[1, 2, 4, 8],
    tile_dimensions=block_max_tile_dimensions,
    use_runtime=[False, True],
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
)
def test_perf_reduce_block_max(
    perf_report,
    formats,
    block_ct_dim,
    tile_dimensions,
    use_runtime,
    run_types,
    loop_factor,
):
    _run_reduce_block_max(
        formats,
        block_ct_dim,
        use_runtime=use_runtime,
        tile_dimensions=tile_dimensions,
        is_perf=True,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
