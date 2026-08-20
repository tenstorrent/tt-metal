# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.llk_params import (
    PERF_LOOP_FACTOR_QUASAR,
    PERF_RUN_TYPES_QUASAR,
    DataFormat,
    DestAccumulation,
    ImpliedMathFormat,
)
from helpers.param_config import parametrize, runtime
from quasar.test_eltwise_binary_sfpu_quasar import (
    _BF16_ADD_SUB_OPS,
    _FLOAT_OPS,
    _INT_OPS,
    _QUANT_OPS,
    DEFAULT_SFPU_BINARY_TILE_INDICES,
    SFPU_BINARY_MAX_MIN_FLOAT_FORMATS,
    SFPU_BINARY_MAX_MIN_INT32_FORMATS,
    _generate_max_min_combinations,
    _get_valid_float_formats_dest_acc,
    max_min_float_dest_acc_for_format,
    max_min_int32_dest_acc_for_format,
)
from quasar.test_eltwise_binary_sfpu_quasar import (
    test_eltwise_binary_sfpu_bf16_rne_quasar as run_eltwise_binary_sfpu_bf16_rne_quasar,
)
from quasar.test_eltwise_binary_sfpu_quasar import (
    test_eltwise_binary_sfpu_float_quasar as run_eltwise_binary_sfpu_float_quasar,
)
from quasar.test_eltwise_binary_sfpu_quasar import (
    test_eltwise_binary_sfpu_int_quasar as run_eltwise_binary_sfpu_int_quasar,
)
from quasar.test_eltwise_binary_sfpu_quasar import (
    test_eltwise_binary_sfpu_max_min_float_quasar as run_eltwise_binary_sfpu_max_min_float_quasar,
)
from quasar.test_eltwise_binary_sfpu_quasar import (
    test_eltwise_binary_sfpu_max_min_int32_quasar as run_eltwise_binary_sfpu_max_min_int32_quasar,
)
from quasar.test_eltwise_binary_sfpu_quasar import (
    test_eltwise_binary_sfpu_quant_quasar as run_eltwise_binary_sfpu_quant_quasar,
)


def _perf_kwargs(perf_report):
    return {
        "run_types": PERF_RUN_TYPES_QUASAR[0],
        "loop_factor": PERF_LOOP_FACTOR_QUASAR,
        "is_perf": True,
        "perf_report": perf_report,
    }


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    binary_op_mathop_clamp_inputs=_INT_OPS,
    data_format_dest_acc=[(DataFormat.Int32, DestAccumulation.Yes)],
    tile_indices=runtime([DEFAULT_SFPU_BINARY_TILE_INDICES]),
)
def test_perf_eltwise_binary_sfpu_int_quasar(
    perf_report,
    binary_op_mathop_clamp_inputs,
    data_format_dest_acc,
    tile_indices,
):
    binary_op, mathop, clamp_inputs = binary_op_mathop_clamp_inputs
    data_format, dest_acc = data_format_dest_acc
    run_eltwise_binary_sfpu_int_quasar(
        data_format,
        dest_acc,
        binary_op,
        mathop,
        clamp_inputs,
        tile_indices,
        **_perf_kwargs(perf_report),
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    binary_op_mathop=_FLOAT_OPS,
    formats_dest_acc=_get_valid_float_formats_dest_acc(),
    implied_math_format=[ImpliedMathFormat.Yes],
    tile_indices=runtime([DEFAULT_SFPU_BINARY_TILE_INDICES]),
)
def test_perf_eltwise_binary_sfpu_float_quasar(
    perf_report,
    formats_dest_acc,
    implied_math_format,
    tile_indices,
    binary_op_mathop,
):
    binary_op, mathop, approx_mode = binary_op_mathop
    run_eltwise_binary_sfpu_float_quasar(
        formats_dest_acc,
        implied_math_format,
        tile_indices,
        binary_op,
        mathop,
        approx_mode,
        **_perf_kwargs(perf_report),
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    binary_op_mathop=_BF16_ADD_SUB_OPS,
    tile_indices=runtime([DEFAULT_SFPU_BINARY_TILE_INDICES]),
)
def test_perf_eltwise_binary_sfpu_bf16_rne_quasar(
    perf_report,
    tile_indices,
    binary_op_mathop,
):
    binary_op, mathop = binary_op_mathop
    run_eltwise_binary_sfpu_bf16_rne_quasar(
        tile_indices,
        binary_op,
        mathop,
        **_perf_kwargs(perf_report),
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats_dest_acc_implied_math_is_max_input_dims=_generate_max_min_combinations(
        SFPU_BINARY_MAX_MIN_FLOAT_FORMATS,
        dest_acc_for_format=max_min_float_dest_acc_for_format,
        implied_math_formats=(ImpliedMathFormat.Yes,),
        input_dimensions_list=([32, 32],),
    ),
    tile_indices=runtime([DEFAULT_SFPU_BINARY_TILE_INDICES]),
)
def test_perf_eltwise_binary_sfpu_max_min_float_quasar(
    perf_report,
    formats_dest_acc_implied_math_is_max_input_dims,
    tile_indices,
):
    run_eltwise_binary_sfpu_max_min_float_quasar(
        formats_dest_acc_implied_math_is_max_input_dims,
        tile_indices,
        **_perf_kwargs(perf_report),
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats_dest_acc_implied_math_is_max_input_dims=_generate_max_min_combinations(
        SFPU_BINARY_MAX_MIN_INT32_FORMATS,
        dest_acc_for_format=max_min_int32_dest_acc_for_format,
        implied_math_formats=(ImpliedMathFormat.No,),
        input_dimensions_list=([32, 32],),
    ),
    tile_indices=runtime([DEFAULT_SFPU_BINARY_TILE_INDICES]),
)
def test_perf_eltwise_binary_sfpu_max_min_int32_quasar(
    perf_report,
    formats_dest_acc_implied_math_is_max_input_dims,
    tile_indices,
):
    run_eltwise_binary_sfpu_max_min_int32_quasar(
        formats_dest_acc_implied_math_is_max_input_dims,
        tile_indices,
        **_perf_kwargs(perf_report),
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    binary_op=_QUANT_OPS,
    sign_magnitude=[False, True],
    tile_indices=runtime([DEFAULT_SFPU_BINARY_TILE_INDICES]),
)
def test_perf_eltwise_binary_sfpu_quant_quasar(
    perf_report,
    binary_op,
    sign_magnitude,
    tile_indices,
):
    run_eltwise_binary_sfpu_quant_quasar(
        binary_op,
        sign_magnitude,
        tile_indices,
        **_perf_kwargs(perf_report),
    )
