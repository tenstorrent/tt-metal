# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import blackhole_only
from helpers.llk_params import MathFidelity
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_sdpa_bcast_col_srca_srcb_reuse import (
    SDPA_FORMATS,
)
from test_sdpa_bcast_col_srca_srcb_reuse import (
    test_sdpa_bcast_col_srca_srcb_reuse as _test_sdpa_bcast_col_srca_srcb_reuse,
)


@pytest.mark.perf
@blackhole_only
@parametrize(
    formats=SDPA_FORMATS,
    math_fidelity=list(MathFidelity),
    **PERF_MEASUREMENT_AXES,
)
def test_perf_sdpa_bcast_col_srca_srcb_reuse(
    perf_report, formats, math_fidelity, run_types, loop_factor, is_perf
):
    # cancellation only rewrites stimuli, so it is not a second kernel measurement.
    _test_sdpa_bcast_col_srca_srcb_reuse(
        formats,
        math_fidelity,
        False,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
