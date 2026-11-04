# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import blackhole_only, skip_for_coverage
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation
from helpers.param_config import input_output_formats, parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_sdpa_weighted_reduce import (
    FILL_CONSTANTS,
)
from test_sdpa_weighted_reduce import (
    test_sdpa_weighted_reduce as _test_sdpa_weighted_reduce,
)

pytestmark = blackhole_only


@pytest.mark.perf
@skip_for_coverage
@parametrize(
    formats=input_output_formats(
        [DataFormat.Float16_b],
        same=True,
    ),
    dest_acc=[DestAccumulation.No],
    # fill_constants only rewrites stimuli; both pairs compile the same kernel.
    fill_constants=[FILL_CONSTANTS[0]],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_sdpa_weighted_reduce(
    perf_report, formats, dest_acc, fill_constants, run_types, loop_factor, is_perf
):
    _test_sdpa_weighted_reduce(
        formats,
        dest_acc,
        fill_constants,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
