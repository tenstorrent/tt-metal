# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import blackhole_only
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_sdpa_bcast_col_srcb_reuse import (
    SDPA_BCAST_COL_SRCB_REUSE_SWEEP,
    _run_sdpa_bcast_col_srcb_reuse,
)


@pytest.mark.perf
@blackhole_only
@parametrize(**SDPA_BCAST_COL_SRCB_REUSE_SWEEP)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_sdpa_bcast_col_srcb_reuse(
    perf_report, formats, mathop, num_tiles, run_types, loop_factor, is_perf
):
    _run_sdpa_bcast_col_srcb_reuse(
        formats,
        mathop,
        num_tiles,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
