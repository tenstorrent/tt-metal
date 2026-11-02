# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Q-head crossing, sanity subset: one small case that fits Wormhole and Blackhole grids. Every crossing case, the
calls that rule crossing out and the larger shapes run in
tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_gqa_crossing_sweep.py."""

import pytest

import ttnn
from tests.ttnn.unit_tests.operations.sdpa.gqa_crossing_test_utils import (
    check_concat_wrap_non_streaming,
    check_gqa_crossing,
)

# 16/4 heads S256 d64, q_chunk 192 on 6x8: a chunk runs on into the next query head of its group
CASE = (2, 16, 4, 256, 64, 192, 256, (6, 8))


@pytest.mark.timeout(300)
def test_sdpa_gqa_crossing(device):
    check_gqa_crossing(device, *CASE, ttnn.bfloat8_b)


@pytest.mark.timeout(300)
def test_sdpa_gqa_crossing_concat_wrap_non_streaming(device):
    check_concat_wrap_non_streaming(device, *CASE)
