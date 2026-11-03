# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import blackhole_only
from helpers.llk_params import MathOperation
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from helpers.sdpa_bcast_utils import (
    SDPA_BCAST_FORMATS,
    SDPA_BCAST_TILE_COUNTS,
    run_sdpa_bcast_col_srcb_reuse,
)


@pytest.mark.perf
@blackhole_only
@parametrize(
    formats=SDPA_BCAST_FORMATS,
    mathop=[MathOperation.Elwmul, MathOperation.Elwadd, MathOperation.Elwsub],
    num_tiles=SDPA_BCAST_TILE_COUNTS,
    **PERF_MEASUREMENT_AXES,
)
def test_perf_sdpa_bcast_col_srcb_reuse(
    perf_report, formats, mathop, num_tiles, run_types, loop_factor, is_perf
):
    run_sdpa_bcast_col_srcb_reuse(
        "sources/sdpa_bcast_col_srcb_reuse_test.cpp",
        formats,
        mathop,
        num_tiles=num_tiles,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
