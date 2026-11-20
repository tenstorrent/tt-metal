# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import test_unpack_transpose as _func
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES


@pytest.mark.perf
@parametrize(
    **_func.UNPACK_TRANSPOSE_SWEEP,
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_unpack_transpose(
    perf_report,
    formats,
    unpack_transpose_faces,
    unpack_transpose_within_face,
    input_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    _func._run_unpack_transpose(
        formats,
        unpack_transpose_faces,
        unpack_transpose_within_face,
        input_dimensions,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
