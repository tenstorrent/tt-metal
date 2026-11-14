# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import blackhole_only
from helpers.llk_params import DestAccumulation
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_custom_mm_reuse_dest_srcb import (
    REUSE_FORMATS,
    REUSE_KT_NT,
)
from test_custom_mm_reuse_dest_srcb import (
    test_custom_mm_reuse_dest_srcb as _test_custom_mm_reuse_dest_srcb,
)


@pytest.mark.perf
@blackhole_only
@parametrize(
    formats=REUSE_FORMATS,
    dest_acc=[DestAccumulation.No],
    reuse_kt_nt=REUSE_KT_NT,
    **PERF_MEASUREMENT_AXES,
)
def test_perf_custom_mm_reuse_dest_srcb(
    perf_report, formats, dest_acc, reuse_kt_nt, run_types, loop_factor, is_perf
):
    _test_custom_mm_reuse_dest_srcb(
        formats,
        dest_acc,
        reuse_kt_nt,
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )
