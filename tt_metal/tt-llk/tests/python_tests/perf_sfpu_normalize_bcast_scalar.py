# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# SFPU scalar normalize, in place, with constant and runtime (compiler-opaque) dest tile indices.

import pytest
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_sfpu_normalize_bcast_scalar import run_norm_scalar


@pytest.mark.perf
@parametrize(
    case=["F32-A_in-rand-rt0", "F32-A_in-rand-rt1"],
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[16],
)
def test_perf_sfpu_normalize_bcast_scalar(perf_report, case, run_types, loop_factor):
    run_norm_scalar(
        case,
        is_perf=True,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
