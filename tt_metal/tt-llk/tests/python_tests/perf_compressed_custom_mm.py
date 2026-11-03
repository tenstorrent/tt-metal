# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import blackhole_only
from helpers.custom_mm_utils import matmul_grid
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_compressed_custom_mm import (
    COMPRESSED_MM_FORMATS,
    CORNER_GRID,
    _run_compressed_custom_mm,
)


def _perf_kwargs(perf_report, run_types, loop_factor, is_perf, case_set):
    return dict(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
        perf_case_set=case_set,
    )


@pytest.mark.perf
@blackhole_only
@parametrize(
    formats=COMPRESSED_MM_FORMATS,
    ct_kt_rows=CORNER_GRID,
    **PERF_MEASUREMENT_AXES,
)
def test_perf_compressed_custom_mm(
    perf_report, formats, ct_kt_rows, run_types, loop_factor, is_perf
):
    _run_compressed_custom_mm(
        formats,
        ct_kt_rows,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf, 0),
    )


@pytest.mark.perf
@blackhole_only
@pytest.mark.nightly
@parametrize(
    formats=COMPRESSED_MM_FORMATS,
    ct_kt_rows=matmul_grid(),
    **PERF_MEASUREMENT_AXES,
)
def test_perf_compressed_custom_mm_full_grid(
    perf_report, formats, ct_kt_rows, run_types, loop_factor, is_perf
):
    _run_compressed_custom_mm(
        formats,
        ct_kt_rows,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf, 1),
    )
