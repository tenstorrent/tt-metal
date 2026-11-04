# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import blackhole_only, skip_for_coverage
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, ReducePool
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES_WITHOUT_MATH_ISOLATE
from test_sdpa_reduce_row import FILL_CONSTANTS
from test_sdpa_reduce_row import test_sdpa_reduce_row as _test_sdpa_reduce_row

pytestmark = blackhole_only


@pytest.mark.perf
@skip_for_coverage
@parametrize(
    formats=input_output_formats(
        [DataFormat.Float16_b],
        same=True,
    ),
    dest_acc=[DestAccumulation.No],
    reduce_pool=[ReducePool.Max],
    # fill_constant is stimulus-only; reduce_pool is the recorded axis.
    fill_constant=[FILL_CONSTANTS[0]],
    **PERF_MEASUREMENT_AXES_WITHOUT_MATH_ISOLATE,
)
def test_perf_sdpa_reduce_row(
    perf_report,
    formats,
    dest_acc,
    reduce_pool,
    fill_constant,
    run_types,
    loop_factor,
    is_perf,
):
    _test_sdpa_reduce_row(
        formats,
        dest_acc,
        reduce_pool,
        fill_constant,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
