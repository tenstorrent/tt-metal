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
    _FLOAT_OPS,
    _INT_OPS,
    DEFAULT_SFPU_BINARY_TILE_INDICES,
    SFPU_BINARY_MAX_MIN_FLOAT_FORMATS,
    SFPU_BINARY_MAX_MIN_INT32_FORMATS,
    _generate_max_min_combinations,
    _get_valid_float_formats_dest_acc,
    max_min_float_dest_acc_for_format,
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


class PerfCaseFamily(Enum):
    INT = "int"
    FLOAT = "float"
    MAX_MIN_FLOAT = "max_min_float"
    MAX_MIN_INT = "max_min_int"


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

    for binary_op, mathop in _FLOAT_OPS:
        for formats, dest_acc in _get_valid_float_formats_dest_acc():
            cases.append(
                pytest.param(
                    (PerfCaseFamily.FLOAT, ((formats, dest_acc), binary_op, mathop)),
                    id=(
                        f"float-{binary_op.lower()}-"
                        f"{formats.input_format.name}-{formats.output_format.name}-"
                        f"{dest_acc.name}"
                    ),
                )
            )

    max_min_float_combinations = _generate_max_min_combinations(
        SFPU_BINARY_MAX_MIN_FLOAT_FORMATS,
        dest_acc_for_format=max_min_float_dest_acc_for_format,
        implied_math_formats=(ImpliedMathFormat.Yes,),
        input_dimensions_list=([32, 32],),
    )
    for combination in max_min_float_combinations:
        formats, dest_acc, _, is_max_op, _ = combination
        op = "max" if is_max_op else "min"
        cases.append(
            pytest.param(
                (PerfCaseFamily.MAX_MIN_FLOAT, combination),
                id=(
                    f"float-{op}-{formats.input_format.name}-"
                    f"{formats.output_format.name}-{dest_acc.name}"
                ),
            )
        )

    max_min_int_combinations = _generate_max_min_combinations(
        SFPU_BINARY_MAX_MIN_INT32_FORMATS,
        dest_acc_for_format=max_min_int32_dest_acc_for_format,
        implied_math_formats=(ImpliedMathFormat.No,),
        input_dimensions_list=([32, 32],),
    )
    for combination in max_min_int_combinations:
        _, _, _, is_max_op, _ = combination
        op = "max" if is_max_op else "min"
        cases.append(
            pytest.param((PerfCaseFamily.MAX_MIN_INT, combination), id=f"int-{op}")
        )

    return cases


@pytest.mark.perf
@pytest.mark.quasar
# The four families have heterogeneous argument shapes and hand-authored IDs;
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
        formats_dest_acc, binary_op, mathop = args
        run_eltwise_binary_sfpu_float_quasar(
            formats_dest_acc,
            ImpliedMathFormat.Yes,
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
    else:
        raise ValueError(f"Unsupported binary SFPU perf family: {family}")
