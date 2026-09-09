# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

import pytest
from helpers.llk_params import (
    PERF_LOOP_FACTOR_QUASAR,
    PERF_RUN_TYPES_QUASAR,
    DataFormat,
    DestAccumulation,
    ImpliedMathFormat,
)
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


class PerfCaseFamily(Enum):
    INT = "int"
    FLOAT = "float"
    BF16_RNE = "bf16_rne"
    MAX_MIN_FLOAT = "max_min_float"
    MAX_MIN_INT = "max_min_int"
    QUANT = "quant"


def _generate_perf_cases():
    cases = []

    for binary_op, mathop, clamp_inputs in _INT_OPS:
        cases.append(
            pytest.param(
                (
                    PerfCaseFamily.INT,
                    (
                        DataFormat.Int32,
                        DestAccumulation.Yes,
                        binary_op,
                        mathop,
                        clamp_inputs,
                    ),
                ),
                id=f"int-{binary_op.lower()}",
            )
        )

    for binary_op, mathop, approx_mode in _FLOAT_OPS:
        for format_variant in _get_valid_float_formats_dest_acc():
            formats = format_variant.formats
            cases.append(
                pytest.param(
                    (
                        PerfCaseFamily.FLOAT,
                        (format_variant, binary_op, mathop, approx_mode),
                    ),
                    id=(
                        f"float-{binary_op.lower()}-{approx_mode.name.lower()}-"
                        f"{formats.input_format.name}-{formats.output_format.name}-"
                        f"{format_variant.dest_acc.name}"
                    ),
                )
            )

    for binary_op, mathop in _BF16_ADD_SUB_OPS:
        cases.append(
            pytest.param(
                (PerfCaseFamily.BF16_RNE, (binary_op, mathop)),
                id=f"bf16-rne-{binary_op.lower()}",
            )
        )

    max_min_float_combinations = _generate_max_min_combinations(
        SFPU_BINARY_MAX_MIN_FLOAT_FORMATS,
        implied_math_formats=(ImpliedMathFormat.Yes,),
        input_dimensions_list=([32, 32],),
    )
    for combination in max_min_float_combinations:
        format_variant, _, is_max_op, _ = combination
        formats = format_variant.formats
        op = "max" if is_max_op else "min"
        cases.append(
            pytest.param(
                (PerfCaseFamily.MAX_MIN_FLOAT, combination),
                id=(
                    f"float-{op}-{formats.input_format.name}-"
                    f"{formats.output_format.name}-{format_variant.dest_acc.name}"
                ),
            )
        )

    max_min_int_combinations = _generate_max_min_combinations(
        SFPU_BINARY_MAX_MIN_INT32_FORMATS,
        implied_math_formats=(ImpliedMathFormat.No,),
        input_dimensions_list=([32, 32],),
    )
    for combination in max_min_int_combinations:
        _, _, is_max_op, _ = combination
        op = "max" if is_max_op else "min"
        cases.append(
            pytest.param((PerfCaseFamily.MAX_MIN_INT, combination), id=f"int-{op}")
        )

    for binary_op in _QUANT_OPS:
        for sign_magnitude in (False, True):
            cases.append(
                pytest.param(
                    (PerfCaseFamily.QUANT, (binary_op, sign_magnitude)),
                    id=f"quant-{binary_op.lower()}-sign-magnitude-{sign_magnitude}",
                )
            )

    return cases


@pytest.mark.perf
@pytest.mark.quasar
# The families have heterogeneous argument shapes and hand-authored IDs;
# plain pytest parametrization keeps them in one homogeneous perf-report module.
@pytest.mark.parametrize("family_and_args", _generate_perf_cases())
def test_perf_eltwise_binary_sfpu_quasar(perf_report, family_and_args):
    family, args = family_and_args
    perf_kwargs = {
        "run_types": PERF_RUN_TYPES_QUASAR[0],
        "loop_factor": PERF_LOOP_FACTOR_QUASAR,
        "is_perf": True,
        "perf_report": perf_report,
    }

    if family == PerfCaseFamily.INT:
        data_format, dest_acc, binary_op, mathop, clamp_inputs = args
        run_eltwise_binary_sfpu_int_quasar(
            data_format,
            dest_acc,
            binary_op,
            mathop,
            clamp_inputs,
            DEFAULT_SFPU_BINARY_TILE_INDICES,
            **perf_kwargs,
        )
    elif family == PerfCaseFamily.FLOAT:
        formats_dest_acc, binary_op, mathop, approx_mode = args
        run_eltwise_binary_sfpu_float_quasar(
            formats_dest_acc,
            ImpliedMathFormat.Yes,
            DEFAULT_SFPU_BINARY_TILE_INDICES,
            binary_op,
            mathop,
            approx_mode,
            **perf_kwargs,
        )
    elif family == PerfCaseFamily.BF16_RNE:
        binary_op, mathop = args
        run_eltwise_binary_sfpu_bf16_rne_quasar(
            DEFAULT_SFPU_BINARY_TILE_INDICES,
            binary_op,
            mathop,
            **perf_kwargs,
        )
    elif family == PerfCaseFamily.MAX_MIN_FLOAT:
        run_eltwise_binary_sfpu_max_min_float_quasar(
            args,
            DEFAULT_SFPU_BINARY_TILE_INDICES,
            **perf_kwargs,
        )
    elif family == PerfCaseFamily.MAX_MIN_INT:
        run_eltwise_binary_sfpu_max_min_int32_quasar(
            args,
            DEFAULT_SFPU_BINARY_TILE_INDICES,
            **perf_kwargs,
        )
    elif family == PerfCaseFamily.QUANT:
        binary_op, sign_magnitude = args
        run_eltwise_binary_sfpu_quant_quasar(
            binary_op,
            sign_magnitude,
            DEFAULT_SFPU_BINARY_TILE_INDICES,
            **perf_kwargs,
        )
    else:
        raise ValueError(f"Unsupported binary SFPU perf family: {family}")
