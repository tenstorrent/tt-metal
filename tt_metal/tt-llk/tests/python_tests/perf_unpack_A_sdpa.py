# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import blackhole_only
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_unpack_A_sdpa import all_params, param_ids
from test_unpack_A_sdpa import test_unpack_A_sdpa as _test_unpack_A_sdpa


@pytest.mark.perf
@blackhole_only
@pytest.mark.parametrize(
    "testname, formats, num_faces, face_r_dim, input_dimensions",
    all_params,
    ids=param_ids,
)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_unpack_A_sdpa(
    perf_report,
    testname,
    formats,
    num_faces,
    face_r_dim,
    input_dimensions,
    run_types,
    loop_factor,
    is_perf,
):
    _test_unpack_A_sdpa(
        testname,
        formats,
        num_faces,
        face_r_dim,
        input_dimensions,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
