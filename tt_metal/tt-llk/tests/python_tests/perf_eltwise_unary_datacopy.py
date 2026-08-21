# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import Tilize
from helpers.param_config import parametrize
from helpers.perf.core import ALL_PERF_RUN_TYPES
from test_eltwise_unary_datacopy import (
    PERF_DATACOPY_COMBINATIONS,
    _run_unary_datacopy_test,
)


@pytest.mark.perf
@parametrize(
    formats_dest_acc_dims=PERF_DATACOPY_COMBINATIONS,
    run_types=[ALL_PERF_RUN_TYPES],
    loop_factor=[32],
    is_perf=[True],
)
def test_perf_eltwise_unary_datacopy(
    perf_report,
    formats_dest_acc_dims,
    run_types,
    loop_factor,
    is_perf,
):
    formats, dest_acc, input_dimensions = formats_dest_acc_dims
    _run_unary_datacopy_test(
        formats,
        dest_acc,
        num_faces=4,
        tilize=Tilize.No,
        input_dimensions=input_dimensions,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
