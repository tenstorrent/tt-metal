# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_eltwise_unary_datacopy import (
    DATACOPY_SUB_BYTE_SWEEP,
    DATACOPY_SWEEP,
    _run_unary_datacopy_test,
)


@pytest.mark.perf
@parametrize(
    **DATACOPY_SWEEP,
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_eltwise_unary_datacopy(
    perf_report,
    formats,
    dest_acc,
    num_faces,
    tilize,
    input_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    _run_unary_datacopy_test(
        formats,
        dest_acc,
        num_faces,
        tilize,
        input_dimensions,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )


@pytest.mark.perf
@parametrize(
    **DATACOPY_SUB_BYTE_SWEEP,
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_eltwise_unary_datacopy_sub_byte_bfp(
    perf_report,
    formats,
    dest_acc,
    num_faces,
    tilize,
    input_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    _run_unary_datacopy_test(
        formats,
        dest_acc,
        num_faces,
        tilize,
        input_dimensions,
        quantize_golden_input=True,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
