# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Quasar tests for binary kernels whose Blackhole implementation uses SFPI.

Each variant selects a ported operation through the normal Quasar
unpack/SFPU/pack harness and compares it with the shared LLK binary golden.
"""

from dataclasses import dataclass
from itertools import product

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    ImpliedMathFormat,
    MathOperation,
    format_dict,
)
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_variant_parameters import APPROX_MODE
from quasar.test_eltwise_binary_sfpu_quasar import (
    _TILE_INDEX_VARIANTS,
    _run_sfpu_binary_llk_golden,
    _stage_binary_operands,
)


@dataclass(frozen=True)
class SfpiBinaryCase:
    mathop: MathOperation
    kernel: str
    function: str
    template_args: tuple[str, ...]
    init: str = "(void)0"
    runtime_args: tuple[str, ...] = ()
    integer: bool = False
    formats: tuple[tuple[InputOutputFormat, DestAccumulation], ...] = ()

    def __repr__(self) -> str:
        return f"{self.kernel}:{self.mathop.name}"


_FLOAT_FORMATS = (
    (InputOutputFormat(DataFormat.Float16, DataFormat.Float16), DestAccumulation.No),
    (
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.No,
    ),
    (
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.Yes,
    ),
    (InputOutputFormat(DataFormat.Float32, DataFormat.Float32), DestAccumulation.Yes),
)
_FP32_FORMATS = _FLOAT_FORMATS[1:]
_INT32_FORMATS = (
    (InputOutputFormat(DataFormat.Int32, DataFormat.Int32), DestAccumulation.Yes),
)


def _case(
    mathop,
    kernel,
    function,
    *template_args,
    init="(void)0",
    runtime_args=(),
    integer=False,
    formats=None,
):
    return SfpiBinaryCase(
        mathop=mathop,
        kernel=kernel,
        function=function,
        template_args=tuple(template_args),
        init=init,
        runtime_args=tuple(runtime_args),
        integer=integer,
        formats=tuple(
            _INT32_FORMATS
            if integer
            else _FLOAT_FORMATS if formats is None else formats
        ),
    )


# One entry per public binary operation.  The three bitwise operations share
# one production header and deliberately remain separate functional variants.
SFPI_BINARY_CASES = (
    _case(
        MathOperation.SfpuAtan2,
        "atan2",
        "calculate_sfpu_atan2",
        "APPROX_MODE",
        "SFPU_ITERATIONS",
        "is_fp32_dest_acc_en",
        init="calculate_sfpu_atan2_init<APPROX_MODE, is_fp32_dest_acc_en>()",
        formats=_FP32_FORMATS,
    ),
    _case(
        MathOperation.SfpuBitwiseAnd,
        "binary_bitwise",
        "calculate_sfpu_binary_bitwise",
        "APPROX_MODE",
        "BinaryBitwiseOp::AND",
        "ckernel::InstrModLoadStore::INT32",
        "SFPU_ITERATIONS",
        integer=True,
    ),
    _case(
        MathOperation.SfpuBitwiseOr,
        "binary_bitwise",
        "calculate_sfpu_binary_bitwise",
        "APPROX_MODE",
        "BinaryBitwiseOp::OR",
        "ckernel::InstrModLoadStore::INT32",
        "SFPU_ITERATIONS",
        integer=True,
    ),
    _case(
        MathOperation.SfpuBitwiseXor,
        "binary_bitwise",
        "calculate_sfpu_binary_bitwise",
        "APPROX_MODE",
        "BinaryBitwiseOp::XOR",
        "ckernel::InstrModLoadStore::INT32",
        "SFPU_ITERATIONS",
        integer=True,
    ),
    _case(
        MathOperation.SfpuBinaryFmod,
        "binary_fmod",
        "calculate_sfpu_binary_fmod",
        "APPROX_MODE",
        "SFPU_ITERATIONS",
        "is_fp32_dest_acc_en",
        init="fmod_binary_init<APPROX_MODE>()",
    ),
    _case(
        MathOperation.SfpuElwpow,
        "binary_pow",
        "calculate_sfpu_binary_pow",
        "APPROX_MODE",
        "SFPU_ITERATIONS",
        "is_fp32_dest_acc_en",
        init="sfpu_binary_pow_init<APPROX_MODE>()",
    ),
    _case(
        MathOperation.SfpuBinaryRemainder,
        "binary_remainder",
        "calculate_sfpu_binary_remainder",
        "APPROX_MODE",
        "SFPU_ITERATIONS",
        "is_fp32_dest_acc_en",
        init="remainder_binary_init<APPROX_MODE, is_fp32_dest_acc_en>()",
    ),
    _case(
        MathOperation.SfpuDivInt32,
        "div_int32",
        "calculate_div_int32",
        "APPROX_MODE",
        "SFPU_ITERATIONS",
        init="div_init<APPROX_MODE>()",
        integer=True,
    ),
    _case(
        MathOperation.SfpuDivInt32Floor,
        "div_int32_floor",
        "calculate_div_int32_floor",
        "APPROX_MODE",
        "SFPU_ITERATIONS",
        init="div_floor_init<APPROX_MODE>()",
        integer=True,
    ),
    _case(
        MathOperation.SfpuIsclose,
        "isclose",
        "calculate_sfpu_isclose",
        "APPROX_MODE",
        "SFPU_ITERATIONS",
        "false",
        init="isclose_init()",
        runtime_args=("0x3727c5acu", "0x322bcc77u"),
        formats=_FP32_FORMATS,
    ),
    _case(
        MathOperation.SfpuLogsigmoid,
        "logsigmoid",
        "calculate_logsigmoid",
        "APPROX_MODE",
        "SFPU_ITERATIONS",
        init="logsigmoid_init<APPROX_MODE>()",
        formats=_FP32_FORMATS,
    ),
    _case(
        MathOperation.SfpuRsubInt32,
        "rsub_int32",
        "calculate_rsub_int",
        "APPROX_MODE",
        "ckernel::InstrModLoadStore::INT32",
        "SFPU_ITERATIONS",
        integer=True,
    ),
)


def _operand_specs(case):
    if case.integer:
        if case.mathop in (
            MathOperation.SfpuDivInt32,
            MathOperation.SfpuDivInt32Floor,
        ):
            spec = StimuliSpec.uniform(low=1.0, high=10000.0)
        else:
            spec = StimuliSpec.uniform(low=-100000.0, high=100000.0)
        return spec, spec
    if case.mathop == MathOperation.SfpuAtan2:
        spec = StimuliSpec.uniform(low=-5.0, high=5.0)
        return spec, spec
    if case.mathop == MathOperation.SfpuLogsigmoid:
        spec = StimuliSpec.uniform(low=-8.0, high=3.9)
        return spec, spec
    return (
        StimuliSpec.uniform(low=0.25, high=2.0),
        StimuliSpec.uniform(low=0.5, high=3.0),
    )


def _prepare_stimuli(case, formats, _dimensions, src0_idx, src1_idx, _mathop):
    spec_a, spec_b = _operand_specs(case)
    operand_a, _, operand_b, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=[32, 32],
        stimuli_format_B=formats.input_format,
        input_dimensions_B=[32, 32],
        spec_A=spec_a,
        spec_B=spec_b,
    )
    if case.mathop == MathOperation.SfpuIsclose:
        operand_b = operand_a.clone()
        operand_b.flatten()[1::2] += 1
    elif case.mathop == MathOperation.SfpuLogsigmoid:
        operand_b = torch.exp(-operand_a.to(torch.float32)).to(
            format_dict[formats.input_format]
        )

    staged, tile_count = _stage_binary_operands(
        operand_a.flatten(),
        operand_b.flatten(),
        (src0_idx, src1_idx, 0),
        format_dict[formats.input_format],
    )
    return staged, tile_count, torch.zeros_like(staged)


_VARIANTS = [
    (case, formats, dest_acc, implied_math_format, tile_indices)
    for case in SFPI_BINARY_CASES
    for (formats, dest_acc), implied_math_format, tile_indices in product(
        case.formats,
        (ImpliedMathFormat.No, ImpliedMathFormat.Yes),
        _TILE_INDEX_VARIANTS,
    )
]


@pytest.mark.quasar
@pytest.mark.parametrize(
    "case,formats,dest_acc,implied_math_format,tile_indices",
    _VARIANTS,
    ids=lambda value: repr(value) if isinstance(value, SfpiBinaryCase) else None,
)
def test_sfpi_compat_binary_quasar(
    case, formats, dest_acc, implied_math_format, tile_indices
):
    torch.manual_seed(42)
    _run_sfpu_binary_llk_golden(
        formats,
        dest_acc,
        implied_math_format,
        tile_indices,
        case.mathop,
        case.mathop.cpp_enum_value,
        prepare_stimuli=lambda *args: _prepare_stimuli(case, *args),
        extra_templates=(APPROX_MODE(ApproximationMode.No),),
    )
