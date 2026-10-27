# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full Q-head crossing sweep: every shape, chunk and grid where the crossing rule applies, in bf16 and bfp8, and the
calls that rule crossing out. Grids larger than the device are skipped."""

import pytest

import ttnn
from tests.ttnn.unit_tests.operations.sdpa.gqa_crossing_test_utils import (
    CROSSING_CASES,
    check_concat_wrap_non_streaming,
    check_gqa_crossing,
    check_ineligible_call,
)


@pytest.mark.parametrize("b, nh, nkv, s, d, q_chunk, k_chunk, grid", CROSSING_CASES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b])
@pytest.mark.timeout(300)
def test_sdpa_gqa_crossing(device, b, nh, nkv, s, d, q_chunk, k_chunk, grid, dtype):
    check_gqa_crossing(device, b, nh, nkv, s, d, q_chunk, k_chunk, grid, dtype)


@pytest.mark.parametrize("b, nh, nkv, s, d, q_chunk, k_chunk, grid", CROSSING_CASES[:2] + CROSSING_CASES[4:])
@pytest.mark.timeout(300)
def test_sdpa_gqa_crossing_concat_wrap_non_streaming(device, b, nh, nkv, s, d, q_chunk, k_chunk, grid):
    check_concat_wrap_non_streaming(device, b, nh, nkv, s, d, q_chunk, k_chunk, grid)


@pytest.mark.parametrize("kind", ["mask", "sink", "causal"])
def test_sdpa_gqa_crossing_ineligible_calls_stay_correct(device, kind):
    check_ineligible_call(device, kind)
