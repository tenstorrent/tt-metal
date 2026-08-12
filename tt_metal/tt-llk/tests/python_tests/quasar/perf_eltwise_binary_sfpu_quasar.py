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
    _FLOAT_OPS,
    _INT_OPS,
    _QUANT_OPS,
    DEFAULT_SFPU_BINARY_TILE_INDICES,
    SFPU_BINARY_MAX_MIN_FLOAT_FORMATS,
    SFPU_BINARY_MAX_MIN_INT32_FORMATS,
    _get_valid_float_formats_dest_acc,
    max_min_dest_acc_modes,
    max_min_float_dest_acc_for_format,
    max_min_implied_math_formats,
    max_min_int32_dest_acc_for_format,
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


def _perf_kwargs(perf_report, run_types, loop_factor):
    return {
        "run_types": run_types,
        "loop_factor": loop_factor,
        "is_perf": True,
        "perf_report": perf_report,
    }


@pytest.mark.perf
@pytest.mark.quasar
@pytest.mark.parametrize("tile_indices", [DEFAULT_SFPU_BINARY_TILE_INDICES])
@pytest.mark.parametrize(
    "binary_op, mathop, clamp_inputs", _INT_OPS, ids=[op for op, _, _ in _INT_OPS]
)
@pytest.mark.parametrize(
    "data_format, dest_acc", [(DataFormat.Int32, DestAccumulation.Yes)]
)
@pytest.mark.parametrize("run_types", PERF_RUN_TYPES_QUASAR)
@pytest.mark.parametrize("loop_factor", [PERF_LOOP_FACTOR_QUASAR])
def test_perf_eltwise_binary_sfpu_int_quasar(
    perf_report,
    data_format,
    dest_acc,
    binary_op,
    mathop,
    clamp_inputs,
    tile_indices,
    run_types,
    loop_factor,
):
    run_eltwise_binary_sfpu_int_quasar(
        data_format,
        dest_acc,
        binary_op,
        mathop,
        clamp_inputs,
        tile_indices,
        **_perf_kwargs(perf_report, run_types, loop_factor),
    )


@pytest.mark.perf
@pytest.mark.quasar
@pytest.mark.parametrize("tile_indices", [DEFAULT_SFPU_BINARY_TILE_INDICES])
@pytest.mark.parametrize(
    "binary_op, mathop", _FLOAT_OPS, ids=[op for op, _ in _FLOAT_OPS]
)
@pytest.mark.parametrize("formats_dest_acc", _get_valid_float_formats_dest_acc())
@pytest.mark.parametrize("implied_math_format", [ImpliedMathFormat.Yes])
@pytest.mark.parametrize("run_types", PERF_RUN_TYPES_QUASAR)
@pytest.mark.parametrize("loop_factor", [PERF_LOOP_FACTOR_QUASAR])
def test_perf_eltwise_binary_sfpu_float_quasar(
    perf_report,
    formats_dest_acc,
    implied_math_format,
    tile_indices,
    binary_op,
    mathop,
    run_types,
    loop_factor,
):
    run_eltwise_binary_sfpu_float_quasar(
        formats_dest_acc,
        implied_math_format,
        tile_indices,
        binary_op,
        mathop,
        **_perf_kwargs(perf_report, run_types, loop_factor),
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats=SFPU_BINARY_MAX_MIN_FLOAT_FORMATS,
    dest_acc=lambda formats: max_min_dest_acc_modes(
        formats, max_min_float_dest_acc_for_format
    ),
    implied_math_format=lambda formats: max_min_implied_math_formats(
        formats, (ImpliedMathFormat.Yes,)
    ),
    is_max_op=[True, False],
    input_dimensions=runtime([[32, 32]]),
    tile_indices=runtime([DEFAULT_SFPU_BINARY_TILE_INDICES]),
)
@pytest.mark.parametrize("run_types", PERF_RUN_TYPES_QUASAR)
@pytest.mark.parametrize("loop_factor", [PERF_LOOP_FACTOR_QUASAR])
def test_perf_eltwise_binary_sfpu_max_min_float_quasar(
    perf_report,
    formats,
    dest_acc,
    implied_math_format,
    is_max_op,
    input_dimensions,
    tile_indices,
    run_types,
    loop_factor,
):
    run_eltwise_binary_sfpu_max_min_float_quasar(
        formats,
        dest_acc,
        implied_math_format,
        is_max_op,
        input_dimensions,
        tile_indices,
        **_perf_kwargs(perf_report, run_types, loop_factor),
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats=SFPU_BINARY_MAX_MIN_INT32_FORMATS,
    dest_acc=lambda formats: max_min_dest_acc_modes(
        formats, max_min_int32_dest_acc_for_format
    ),
    implied_math_format=lambda formats: max_min_implied_math_formats(
        formats, (ImpliedMathFormat.No,)
    ),
    is_max_op=[True, False],
    input_dimensions=runtime([[32, 32]]),
    tile_indices=runtime([DEFAULT_SFPU_BINARY_TILE_INDICES]),
)
@pytest.mark.parametrize("run_types", PERF_RUN_TYPES_QUASAR)
@pytest.mark.parametrize("loop_factor", [PERF_LOOP_FACTOR_QUASAR])
def test_perf_eltwise_binary_sfpu_max_min_int32_quasar(
    perf_report,
    formats,
    dest_acc,
    implied_math_format,
    is_max_op,
    input_dimensions,
    tile_indices,
    run_types,
    loop_factor,
):
    run_eltwise_binary_sfpu_max_min_int32_quasar(
        formats,
        dest_acc,
        implied_math_format,
        is_max_op,
        input_dimensions,
        tile_indices,
        **_perf_kwargs(perf_report, run_types, loop_factor),
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    binary_op=_QUANT_OPS,
    sign_magnitude=[False, True],
    tile_indices=runtime([DEFAULT_SFPU_BINARY_TILE_INDICES]),
)
@pytest.mark.parametrize("run_types", PERF_RUN_TYPES_QUASAR)
@pytest.mark.parametrize("loop_factor", [PERF_LOOP_FACTOR_QUASAR])
def test_perf_eltwise_binary_sfpu_quant_quasar(
    perf_report,
    binary_op,
    sign_magnitude,
    tile_indices,
    run_types,
    loop_factor,
):
    run_eltwise_binary_sfpu_quant_quasar(
        binary_op,
        sign_magnitude,
        tile_indices,
        **_perf_kwargs(perf_report, run_types, loop_factor),
    )
