# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import quasar.test_eltwise_binary_sfpu_quasar as _func
from helpers.llk_params import (
    PERF_LOOP_FACTOR_QUASAR,
    PERF_RUN_TYPES_QUASAR,
    ApproximationMode,
    MathOperation,
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


def _perf_approx_modes(mathop):
    # Functional sweeps DIV Yes/No. Perf pins No, matching BH/WH, and keeps the
    # atan2 Yes/No pair because that kernel reads APPROX_MODE.
    if mathop == MathOperation.SfpuAtan2:
        return [ApproximationMode.No, ApproximationMode.Yes]
    return [ApproximationMode.No]


_FLOAT_PERF_SWEEP = {
    **_func.FLOAT_SWEEP,
    "approx_mode": _perf_approx_modes,
}


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    **_func.INT_SWEEP,
    approx_mode=[ApproximationMode.No],
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_int_quasar(
    perf_report,
    formats,
    dest_acc,
    mathop,
    sign_magnitude,
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
        sign_magnitude=sign_magnitude,
        approx_mode=approx_mode,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    **_FLOAT_PERF_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_float_quasar(
    perf_report,
    formats,
    dest_acc,
    mathop,
    approx_mode,
    run_types,
    loop_factor,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_float_quasar(
        formats,
        dest_acc,
        mathop,
        approx_mode,
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


_BCAST_PERF_SWEEP = {
    **_func.BCAST_SWEEP,
}


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    **_BCAST_PERF_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_bcast_quasar(
    perf_report,
    formats,
    dest_acc,
    mathop,
    broadcast_type,
    run_types,
    loop_factor,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_bcast_quasar(
        formats,
        dest_acc,
        mathop,
        broadcast_type,
        _func.DEFAULT_SFPU_BINARY_TILE_INDICES,
        **_perf_kwargs(perf_report, run_types, loop_factor, is_perf),
    )


_MAX_MIN_FLOAT_PERF_SWEEP = dict(
    formats_dest_acc_is_max_input_dims=_func._generate_max_min_combinations(
        _func.SFPU_BINARY_MAX_MIN_FLOAT_FORMATS,
    ),
)


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    **_MAX_MIN_FLOAT_PERF_SWEEP,
    **_PERF_AXES,
)
def test_perf_eltwise_binary_sfpu_max_min_float_quasar(
    perf_report,
    formats_dest_acc_is_max_input_dims,
    run_types,
    loop_factor,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_max_min_float_quasar(
        formats_dest_acc_is_max_input_dims,
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
    formats_dest_acc_is_max_input_dims,
    run_types,
    loop_factor,
    is_perf,
):
    _func.test_eltwise_binary_sfpu_max_min_int32_quasar(
        formats_dest_acc_is_max_input_dims,
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
