# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""relu_clamp_int / relu_clamp_uint perf, sharing sources/relu_clamp_int_test.cpp with test_relu_clamp_int.py.

A subset of the functional thresholds: one per kernel path (int32 relu_min with a positive and a
negative threshold, int32 relu_max, uint32 with the threshold top bit clear and set), default build.
"""

import pytest
from conftest import skip_for_blackhole, skip_for_quasar
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_relu_clamp_int import (
    RELU_MAX_INT,
    RELU_MAX_UINT,
    RELU_MIN_INT,
    RELU_MIN_UINT,
    run_relu_clamp,
)

PERF_CASES = {
    "min_int32_pos": (RELU_MIN_INT, 6),
    "min_int32_neg": (RELU_MIN_INT, -6),
    "max_int32_pos": (RELU_MAX_INT, 6),
    "min_uint32_msb_clear": (RELU_MIN_UINT, 6),
    "min_uint32_msb_set": (RELU_MIN_UINT, 2**31 + 6),
    "max_uint32_msb_clear": (RELU_MAX_UINT, 6),
    "max_uint32_msb_set": (RELU_MAX_UINT, 2**31 + 6),
}


@pytest.mark.perf
@skip_for_blackhole
@skip_for_quasar
@parametrize(
    case=list(PERF_CASES),
    mm_throttle=[0],
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
)
def test_perf_relu_clamp_int(perf_report, case, mm_throttle, run_types, loop_factor):
    variant, threshold = PERF_CASES[case]
    run_relu_clamp(
        variant,
        threshold,
        mm_throttle,
        is_perf=True,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
