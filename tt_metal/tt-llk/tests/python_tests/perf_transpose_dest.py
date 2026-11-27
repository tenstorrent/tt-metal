# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_transpose_dest import (
    TRANSPOSE_DEST_FLOAT_PERF_SWEEP,
    TRANSPOSE_DEST_INT_SWEEP,
    _run_transpose_dest,
)


@pytest.mark.perf
@parametrize(
    **TRANSPOSE_DEST_FLOAT_PERF_SWEEP,
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_transpose_dest_float(
    perf_report,
    formats,
    dest_acc,
    math_transpose_faces,
    unpack_to_dest,
    run_types,
    loop_factor,
    is_perf,
):
    _run_transpose_dest(
        formats,
        dest_acc=dest_acc,
        math_transpose_faces=math_transpose_faces,
        unpack_to_dest=unpack_to_dest,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )


@pytest.mark.perf
@parametrize(
    **TRANSPOSE_DEST_INT_SWEEP,
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_transpose_dest_int(
    perf_report,
    formats,
    dest_acc,
    math_transpose_faces,
    unpack_to_dest,
    run_types,
    loop_factor,
    is_perf,
):
    _run_transpose_dest(
        formats,
        dest_acc=dest_acc,
        math_transpose_faces=math_transpose_faces,
        unpack_to_dest=unpack_to_dest,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
