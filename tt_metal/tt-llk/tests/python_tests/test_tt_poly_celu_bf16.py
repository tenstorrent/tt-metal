# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Direct generated-BF16 primitive, using the stock unary SFPU device harness."""

from dataclasses import dataclass

from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    FastMode,
    MathOperation,
)
from helpers.test_variant_parameters import TemplateParameter
from test_eltwise_unary_sfpu import eltwise_unary_sfpu


@dataclass
class _GeneratedBF16(TemplateParameter):
    def convert_to_cpp(self) -> str:
        return (
            '#define TT_POLY_LLK_TEST_HEADER "llk_sfpu/ckernel_sfpu_celu.h"\n'
            "#define TT_POLY_LLK_TEST_CALC calculate_celu_tt_poly_bf16\n"
            "#define TT_POLY_LLK_TEST_ITERATIONS 32\n"
            "#define TT_POLY_LLK_TEST_VECTOR_MODE None\n"
        )


def test_tt_poly_celu_bf16_llk():
    eltwise_unary_sfpu(
        "sources/eltwise_unary_sfpu_test.cpp",
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.No,
        ApproximationMode.No,
        MathOperation.Celu,
        FastMode.No,
        [32, 32],
        extra_templates=(_GeneratedBF16(),),
    )
