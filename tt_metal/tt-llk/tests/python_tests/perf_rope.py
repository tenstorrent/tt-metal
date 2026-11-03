# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import blackhole_only
from helpers.llk_params import DestAccumulation
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from test_rope import (
    DENSE_STRIDE,
    TILE_SLOT_STRIDE,
    _heads,
    invoke_with_perf,
    test_rope,
    test_rope_fused_cos_sin,
    test_rope_fused_dest_precision,
    test_rope_nonzero_x_base,
    test_rope_scale,
    test_rope_zero_scale,
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
@parametrize(stride=[TILE_SLOT_STRIDE, DENSE_STRIDE], wt=[1, 2], ht=_heads)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_rope(perf_report, stride, wt, ht, run_types, loop_factor, is_perf):
    invoke_with_perf(
        test_rope,
        stride,
        wt,
        ht,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_rope_nonzero_x_base(perf_report, run_types, loop_factor, is_perf):
    invoke_with_perf(
        test_rope_nonzero_x_base,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_rope_scale(perf_report, run_types, loop_factor, is_perf):
    invoke_with_perf(
        test_rope_scale, **_perf_kwargs(perf_report, run_types, loop_factor, is_perf)
    )


@pytest.mark.perf
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_rope_zero_scale(perf_report, run_types, loop_factor, is_perf):
    invoke_with_perf(
        test_rope_zero_scale,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(
    tile_h=[1, 2, 4, 8, 16, 32],
    # Dense matmul slots hold the top two faces; 32 live rows need a full tile slot.
    x_stride=lambda tile_h: (
        [TILE_SLOT_STRIDE, DENSE_STRIDE] if tile_h <= 16 else [TILE_SLOT_STRIDE]
    ),
    per_row=[False, True],
    scale=[None, 0.0, -2.0],
)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_rope_fused_cos_sin(
    perf_report, tile_h, x_stride, per_row, scale, run_types, loop_factor, is_perf
):
    invoke_with_perf(
        test_rope_fused_cos_sin,
        tile_h,
        x_stride,
        per_row,
        scale,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(
    # (heads, width tiles, x stride, height, per-row phase, scale, phases first)
    case=[
        (2, 1, TILE_SLOT_STRIDE, 1, False, None, False),
        (1, 2, TILE_SLOT_STRIDE, 2, True, -2.0, False),
        (2, 1, DENSE_STRIDE, 4, False, -2.0, False),
        (1, 2, DENSE_STRIDE, 8, True, None, False),
        (1, 2, TILE_SLOT_STRIDE, 16, False, 0.0, False),
        (1, 2, TILE_SLOT_STRIDE, 32, True, None, False),
        (1, 1, TILE_SLOT_STRIDE, 32, False, -2.0, True),
        (1, 2, TILE_SLOT_STRIDE, 32, True, 0.0, False),
    ],
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
)
@parametrize(**PERF_MEASUREMENT_AXES)
def test_perf_rope_fused_dest_precision(
    perf_report, case, dest_acc, run_types, loop_factor, is_perf
):
    invoke_with_perf(
        test_rope_fused_dest_precision,
        case,
        dest_acc,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )
