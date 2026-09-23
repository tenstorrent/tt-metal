# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import quasar.test_eltwise_binary_sfpu_quasar as _func
from helpers.llk_params import (
    PERF_LOOP_FACTOR_QUASAR,
    PERF_RUN_TYPES_QUASAR,
    ApproximationMode,
)
from helpers.param_config import parametrize

_PERF_AXES = dict(
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)


def _perf_kwargs(perf_report, run_types, loop_factor, is_perf):
    return dict(
        is_perf=is_perf,
        perf_report=perf_report,
        run_types=run_types,
        loop_factor=loop_factor,
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    **_func.INT_SWEEP,
    approx_mode=[
        ApproximationMode.Yes,
        ApproximationMode.No,
    ],
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_int_quasar(
    perf_report,
    formats,
    dest_acc,
    mathop,
    approx_mode,
    run_types,
    loop_factor,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_int_quasar(
        formats,
        dest_acc,
        mathop,
        _func.DEFAULT_SFPU_BINARY_TILE_INDICES,
        approx_mode=approx_mode,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    **_func.FLOAT_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_float_quasar(
    perf_report,
    formats,
    dest_acc,
    mathop,
    approx_mode,
    implied_math_format,
    run_types,
    loop_factor,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_float_quasar(
        formats,
        dest_acc,
        mathop,
        approx_mode,
        implied_math_format,
        _func.DEFAULT_SFPU_BINARY_TILE_INDICES,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    **_func.BF16_RNE_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_bf16_rne_quasar(
    perf_report,
    binary_op_mathop,
    run_types,
    loop_factor,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_bf16_rne_quasar(
        binary_op_mathop,
        _func.DEFAULT_SFPU_BINARY_TILE_INDICES,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    **_func.MAX_MIN_FLOAT_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_max_min_float_quasar(
    perf_report,
    formats_dest_acc_implied_math_is_max_input_dims,
    run_types,
    loop_factor,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_max_min_float_quasar(
        formats_dest_acc_implied_math_is_max_input_dims,
        _func.DEFAULT_SFPU_BINARY_TILE_INDICES,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    **_func.MAX_MIN_INT32_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_max_min_int32_quasar(
    perf_report,
    formats_dest_acc_implied_math_is_max_input_dims,
    run_types,
    loop_factor,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_max_min_int32_quasar(
        formats_dest_acc_implied_math_is_max_input_dims,
        _func.DEFAULT_SFPU_BINARY_TILE_INDICES,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    **_func.QUANT_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_quant_quasar(
    perf_report,
    binary_op,
    sign_magnitude,
    run_types,
    loop_factor,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_quant_quasar(
        binary_op,
        sign_magnitude,
        _func.DEFAULT_SFPU_BINARY_TILE_INDICES,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )
