# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.param_config import parametrize
from helpers.perf.blaze import PERF_MEASUREMENT_AXES
from helpers.tile_constants import DEFAULT_TILE_C_DIM, DEFAULT_TILE_R_DIM
from test_sdpa_custom_mm import SHAPES, _run, _skip_on_simulator

pytestmark = [skip_for_wormhole, skip_for_quasar]


def _perf_kwargs(perf_report, run_types, loop_factor, is_perf):
    return dict(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )


def _shape(shape):
    return shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape


@pytest.mark.perf
@parametrize(
    shape=SHAPES,
    **PERF_MEASUREMENT_AXES,
)
def test_perf_sdpa_custom_mm(
    perf_report, request, shape, run_types, loop_factor, is_perf
):
    _skip_on_simulator(request)
    M, K, N = _shape(shape)
    _run(
        M,
        K,
        N,
        signal_granularity=1,
        read_transposed=False,
        mm_transpose=False,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(
    shape=SHAPES,
    **PERF_MEASUREMENT_AXES,
)
def test_perf_sdpa_custom_mm_read_transposed(
    perf_report, request, shape, run_types, loop_factor, is_perf
):
    _skip_on_simulator(request)
    M, K, N = _shape(shape)
    if N // DEFAULT_TILE_C_DIM == 1:
        pytest.skip(
            "read_transposed is a no-op for ct_dim==1 and the LLK's ct==1 fast path is "
            "shaped for the canonical contiguous read; not drivable standalone"
        )
    _run(
        M,
        K,
        N,
        signal_granularity=1,
        read_transposed=True,
        mm_transpose=False,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(
    shape_sg=[
        ((2, 64, 64), 2),
        ((8, 256, 128), 2),
        ((8, 256, 128), 4),
    ],
    **PERF_MEASUREMENT_AXES,
)
def test_perf_sdpa_custom_mm_signal_granularity(
    perf_report, request, shape_sg, run_types, loop_factor, is_perf
):
    _skip_on_simulator(request)
    (shape, sg) = (
        shape_sg[0]
        if len(shape_sg) == 1 and isinstance(shape_sg[0], tuple)
        else shape_sg
    )
    M, K, N = shape
    _run(
        M,
        K,
        N,
        signal_granularity=sg,
        read_transposed=False,
        mm_transpose=False,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@parametrize(
    shape=[(1, 64, 96), (8, 128, 128)],
    stride_mode=["implicit", "zero", "tight", "padded"],
    read_transposed=lambda stride_mode: (
        [False, True] if stride_mode == "padded" else [True]
    ),
    input_tile_offset=lambda stride_mode: ([0, 1] if stride_mode == "padded" else [0]),
    **PERF_MEASUREMENT_AXES,
)
def test_perf_sdpa_custom_mm_row_stride(
    perf_report,
    request,
    shape,
    stride_mode,
    read_transposed,
    input_tile_offset,
    run_types,
    loop_factor,
    is_perf,
):
    _skip_on_simulator(request)
    M, K, N = shape
    kt = K // DEFAULT_TILE_R_DIM
    row_stride = {"implicit": None, "zero": 0, "tight": kt, "padded": kt + 2}[
        stride_mode
    ]
    _run(
        M,
        K,
        N,
        signal_granularity=1,
        read_transposed=read_transposed,
        mm_transpose=False,
        row_stride=row_stride,
        input_tile_offset=input_tile_offset,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )
