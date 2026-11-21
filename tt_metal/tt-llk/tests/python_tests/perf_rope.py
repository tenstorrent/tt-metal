# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import blackhole_only
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_rope import (
    ROPE_FUSED_COS_SIN_SWEEP,
    ROPE_FUSED_DEST_PRECISION_SWEEP,
    ROPE_LAYOUT_SWEEP,
    _run_rope_case,
    _run_rope_fused_cos_sin,
    _run_rope_fused_dest_precision,
    _run_rope_nonzero_x_base,
    _run_rope_scale,
    _run_rope_zero_scale,
)

pytestmark = blackhole_only


def _perf_kwargs(perf_report, run_types, loop_factor, is_perf):
    return dict(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )


@pytest.mark.perf
@parametrize(**ROPE_LAYOUT_SWEEP)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_rope(perf_report, stride, wt, ht, run_types, loop_factor, is_perf):
    _run_rope_case(
        stride,
        wt,
        ht,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_rope_nonzero_x_base(perf_report, run_types, loop_factor, is_perf):
    _run_rope_nonzero_x_base(
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_rope_scale(perf_report, run_types, loop_factor, is_perf):
    _run_rope_scale(**_perf_kwargs(perf_report, run_types, loop_factor, is_perf))


@pytest.mark.perf
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_rope_zero_scale(perf_report, run_types, loop_factor, is_perf):
    _run_rope_zero_scale(
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(**ROPE_FUSED_COS_SIN_SWEEP)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_rope_fused_cos_sin(
    perf_report, tile_h, x_stride, per_row, scale, run_types, loop_factor, is_perf
):
    _run_rope_fused_cos_sin(
        tile_h,
        x_stride,
        per_row,
        scale,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(**ROPE_FUSED_DEST_PRECISION_SWEEP)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_rope_fused_dest_precision(
    perf_report, case, dest_acc, run_types, loop_factor, is_perf
):
    _run_rope_fused_dest_precision(
        case,
        dest_acc,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )
