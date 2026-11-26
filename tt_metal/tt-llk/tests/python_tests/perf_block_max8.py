# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Paired performance coverage for SFPU and FPU block_max8."""

import pytest
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_block_max8 import (
    DST_INDICES,
    PATTERNS,
    PERF_VALID_COUNTS,
    _run_block_max8,
)
from tt_llk_harness import blackhole_only

LOOP_FACTOR = 32


@pytest.mark.perf
@blackhole_only
@pytest.mark.parametrize("dst_index", DST_INDICES)
@pytest.mark.parametrize("valid", PERF_VALID_COUNTS)
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("run_types", [ALL_PERF_RUN_TYPES])
@pytest.mark.parametrize("loop_factor", [LOOP_FACTOR])
@pytest.mark.parametrize("is_perf", [True])
def test_perf_block_max8(
    perf_report, dst_index, valid, pattern, run_types, loop_factor, is_perf
):
    _run_block_max8(
        "sources/block_max8_test.cpp",
        dst_index,
        valid,
        pattern,
        negative_zero_result=-1.0,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )


@pytest.mark.perf
@blackhole_only
@pytest.mark.parametrize("dst_index", DST_INDICES)
@pytest.mark.parametrize("valid", PERF_VALID_COUNTS)
@pytest.mark.parametrize("pattern", PATTERNS + ["denormal"])
@pytest.mark.parametrize("fpu_calls", [1, 3])
@pytest.mark.parametrize("run_types", [ALL_PERF_RUN_TYPES])
@pytest.mark.parametrize("loop_factor", [LOOP_FACTOR])
@pytest.mark.parametrize("is_perf", [True])
def test_perf_block_max8_fpu(
    perf_report,
    dst_index,
    valid,
    pattern,
    fpu_calls,
    run_types,
    loop_factor,
    is_perf,
):
    _run_block_max8(
        "sources/block_max8_fpu_test.cpp",
        dst_index,
        valid,
        pattern,
        negative_zero_result=0.0,
        fpu_calls=fpu_calls,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
