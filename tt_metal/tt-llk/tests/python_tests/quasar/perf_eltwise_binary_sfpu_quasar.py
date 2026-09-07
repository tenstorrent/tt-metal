# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
from helpers.dest_params import (
    UnpackPath,
    dest_acc_modes,
    dest_sync_modes,
    unpack_to_dest_modes,
)
from helpers.format_config import DataFormat
from helpers.llk_params import (
    PERF_LOOP_FACTOR_QUASAR,
    PERF_RUN_TYPES_QUASAR,
    ImpliedMathFormat,
)
from helpers.param_config import input_output_formats, parametrize
from quasar.test_eltwise_binary_sfpu_quasar import (
    _BF16_ADD_SUB_OPS,
    _FLOAT_OPS,
    _INT_OPS,
    _QUANT_OPS,
    DEFAULT_SFPU_BINARY_TILE_INDICES,
    SFPU_BINARY_FLOAT_FORMATS,
    SFPU_BINARY_MAX_MIN_FLOAT_FORMATS,
    SFPU_BINARY_MAX_MIN_INT32_FORMATS,
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


@pytest.mark.perf
@pytest.mark.quasar
@pytest.mark.parametrize(
    "binary_op, mathop, approx_mode",
    _FLOAT_OPS,
    ids=[f"{op}_{approx.name}" for op, _, approx in _FLOAT_OPS],
)
@parametrize(
    formats=SFPU_BINARY_FLOAT_FORMATS,
    dest_acc=dest_acc_modes,
    dest_sync=lambda: dest_sync_modes(is_perf=True),
    unpack_to_dest=lambda formats, dest_acc: unpack_to_dest_modes(
        formats, dest_acc, path=UnpackPath.Sfpu
    ),
    implied_math_format=[ImpliedMathFormat.Yes],
    tile_indices=[DEFAULT_SFPU_BINARY_TILE_INDICES],
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_eltwise_binary_sfpu_float_quasar(
    perf_report,
    formats,
    dest_acc,
    dest_sync,
    unpack_to_dest,
    implied_math_format,
    tile_indices,
    binary_op,
    mathop,
    approx_mode,
    run_types,
    loop_factor,
    is_perf,
):
    run_eltwise_binary_sfpu_float_quasar(
        formats,
        dest_acc,
        dest_sync,
        unpack_to_dest,
        implied_math_format,
        tile_indices,
        binary_op,
        mathop,
        approx_mode,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )


@pytest.mark.perf
@pytest.mark.quasar
@pytest.mark.parametrize(
    "binary_op, mathop, clamp_inputs",
    _INT_OPS,
    ids=[op for op, _, _ in _INT_OPS],
)
@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    dest_acc=dest_acc_modes,
    dest_sync=lambda: dest_sync_modes(is_perf=True),
    unpack_to_dest=lambda formats, dest_acc: unpack_to_dest_modes(
        formats, dest_acc, path=UnpackPath.Int32Dest
    ),
    tile_indices=[DEFAULT_SFPU_BINARY_TILE_INDICES],
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_eltwise_binary_sfpu_int_quasar(
    perf_report,
    formats,
    dest_acc,
    dest_sync,
    unpack_to_dest,
    binary_op,
    mathop,
    clamp_inputs,
    tile_indices,
    run_types,
    loop_factor,
    is_perf,
):
    run_eltwise_binary_sfpu_int_quasar(
        formats,
        dest_acc,
        dest_sync,
        unpack_to_dest,
        binary_op,
        mathop,
        clamp_inputs,
        tile_indices,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )


@pytest.mark.perf
@pytest.mark.quasar
@pytest.mark.parametrize(
    "binary_op, mathop",
    _BF16_ADD_SUB_OPS,
    ids=[op for op, _ in _BF16_ADD_SUB_OPS],
)
@parametrize(
    tile_indices=[DEFAULT_SFPU_BINARY_TILE_INDICES],
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_eltwise_binary_sfpu_bf16_rne_quasar(
    perf_report,
    tile_indices,
    binary_op,
    mathop,
    run_types,
    loop_factor,
    is_perf,
):
    run_eltwise_binary_sfpu_bf16_rne_quasar(
        tile_indices,
        binary_op,
        mathop,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats=SFPU_BINARY_MAX_MIN_FLOAT_FORMATS,
    dest_acc=dest_acc_modes,
    dest_sync=lambda: dest_sync_modes(is_perf=True),
    unpack_to_dest=lambda formats, dest_acc: unpack_to_dest_modes(
        formats, dest_acc, path=UnpackPath.Sfpu
    ),
    implied_math_format=[ImpliedMathFormat.Yes],
    is_max_op=[True, False],
    input_dimensions=[[32, 32]],
    tile_indices=[DEFAULT_SFPU_BINARY_TILE_INDICES],
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_eltwise_binary_sfpu_max_min_float_quasar(
    perf_report,
    formats,
    dest_acc,
    dest_sync,
    unpack_to_dest,
    implied_math_format,
    is_max_op,
    input_dimensions,
    tile_indices,
    run_types,
    loop_factor,
    is_perf,
):
    run_eltwise_binary_sfpu_max_min_float_quasar(
        formats,
        dest_acc,
        dest_sync,
        unpack_to_dest,
        implied_math_format,
        is_max_op,
        input_dimensions,
        tile_indices,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    formats=SFPU_BINARY_MAX_MIN_INT32_FORMATS,
    dest_acc=dest_acc_modes,
    dest_sync=lambda: dest_sync_modes(is_perf=True),
    unpack_to_dest=lambda formats, dest_acc: unpack_to_dest_modes(
        formats, dest_acc, path=UnpackPath.Sfpu
    ),
    is_max_op=[True, False],
    input_dimensions=[[32, 32]],
    tile_indices=[DEFAULT_SFPU_BINARY_TILE_INDICES],
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_eltwise_binary_sfpu_max_min_int32_quasar(
    perf_report,
    formats,
    dest_acc,
    dest_sync,
    unpack_to_dest,
    is_max_op,
    input_dimensions,
    tile_indices,
    run_types,
    loop_factor,
    is_perf,
):
    run_eltwise_binary_sfpu_max_min_int32_quasar(
        formats,
        dest_acc,
        dest_sync,
        unpack_to_dest,
        is_max_op,
        input_dimensions,
        tile_indices,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )


@pytest.mark.perf
@pytest.mark.quasar
@parametrize(
    binary_op=_QUANT_OPS,
    sign_magnitude=[False, True],
    tile_indices=[DEFAULT_SFPU_BINARY_TILE_INDICES],
    run_types=PERF_RUN_TYPES_QUASAR,
    loop_factor=[PERF_LOOP_FACTOR_QUASAR],
    is_perf=[True],
)
def test_perf_eltwise_binary_sfpu_quant_quasar(
    perf_report,
    binary_op,
    sign_magnitude,
    tile_indices,
    run_types,
    loop_factor,
    is_perf,
):
    run_eltwise_binary_sfpu_quant_quasar(
        binary_op,
        sign_magnitude,
        tile_indices,
        run_types=run_types,
        loop_factor=loop_factor,
        is_perf=is_perf,
        perf_report=perf_report,
    )
