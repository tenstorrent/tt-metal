# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import blackhole_only
from helpers.llk_params import MathFidelity
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_sdpa_bcast_col_srca_srcb_reuse import (
    SDPA_FORMATS,
    _run_sdpa_bcast_col_srca_srcb_reuse,
)


@pytest.mark.perf
@blackhole_only
@parametrize(
    formats=SDPA_FORMATS,
    math_fidelity=list(MathFidelity),
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_sdpa_bcast_col_srca_srcb_reuse(
    perf_report,
    formats,
    math_fidelity,
    run_types,
    loop_factor,
    is_perf,
):
    _run_sdpa_bcast_col_srca_srcb_reuse(
        formats,
        math_fidelity,
        False,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
