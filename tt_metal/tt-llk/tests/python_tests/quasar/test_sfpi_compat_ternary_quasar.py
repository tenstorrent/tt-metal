# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Quasar tests for ternary kernels whose Blackhole implementation uses SFPI."""

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TernarySFPUGolden, get_golden_generator
from helpers.llk_params import (
    ApproximationMode,
    DataCopyType,
    DestAccumulation,
    ImpliedMathFormat,
    MathOperation,
    UnpackerEngine,
    VectorMode,
    format_dict,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    MATH_OP,
    NUM_FACES,
    SFPU_TERNARY_SCALAR,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
    VECTOR_MODE,
    TemplateParameter,
)
from helpers.utils import passed_test

_CPP_SOURCE = "sources/quasar/sfpu_where_quasar_test.cpp"
_QSR_SFPU = "../../../../hw/ckernels/quasar/metal/llk_api/llk_sfpu"
_QSR_SFPU_DIR = (
    Path(__file__).resolve().parents[4] / "hw/ckernels/quasar/metal/llk_api/llk_sfpu"
)
_SCALAR_BITS = 0x3FC00000  # 1.5f
_FACE_SIZE = 16 * 16
_PROCESSED_FACES = {
    VectorMode.None_: (0,),
    VectorMode.R: (0, 1),
    VectorMode.C: (0, 2),
    VectorMode.RC: (0, 1, 2, 3),
}
_FORMATS_DEST_ACC = (
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
_ADDCMUL_FORMATS_DEST_ACC = (
    (InputOutputFormat(DataFormat.Float16, DataFormat.Float16), DestAccumulation.No),
    *_FORMATS_DEST_ACC,
)


@dataclass(frozen=True)
class SfpiTernaryCase:
    mathop: MathOperation
    kernel: str
    function: str
    init: str = "(void)0"
    scalar: bool = False
    formats: tuple[tuple[InputOutputFormat, DestAccumulation], ...] = _FORMATS_DEST_ACC

    def __repr__(self) -> str:
        return f"{self.kernel}:{self.mathop.name}"


@dataclass
class SFPI_COMPAT_TERNARY_KERNEL(TemplateParameter):
    case: SfpiTernaryCase
    data_format: DataFormat

    def convert_to_cpp(self) -> str:
        runtime_args = ", SFPU_TERNARY_SCALAR" if self.case.scalar else ""
        call = (
            "SFPU_TERNARY_CALL(dest_sync, is_fp32_dest_acc_en, "
            f"{self.case.function}, (APPROX_MODE, is_fp32_dest_acc_en, "
            f"DataFormat::{self.data_format.name}, SFPU_ITERATIONS), "
            f"in0, in1, in2, out, VECTOR_MODE{runtime_args})"
        )
        return "\n".join(
            [
                "#define SFPI_COMPAT_TEST",
                f'#define SFPI_COMPAT_HEADER "{_QSR_SFPU}/ckernel_sfpu_{self.case.kernel}.h"',
                f"#define SFPI_COMPAT_INIT() {self.case.init}",
                f"#define SFPI_COMPAT_CALL(in0, in1, in2, out) {call}",
            ]
        )


SFPI_TERNARY_CASES = (
    SfpiTernaryCase(
        MathOperation.SfpuAddcdiv,
        "addcdiv",
        "calculate_addcdiv",
        init="init_addcdiv<APPROX_MODE>()",
        scalar=True,
    ),
    SfpiTernaryCase(
        MathOperation.SfpuAddcmul,
        "addcmul",
        "calculate_addcmul",
        scalar=True,
        formats=_ADDCMUL_FORMATS_DEST_ACC,
    ),
    SfpiTernaryCase(MathOperation.SfpuLerp, "lerp", "calculate_lerp"),
    SfpiTernaryCase(
        MathOperation.SfpuSnakeBeta,
        "snake_beta",
        "calculate_snake_beta",
        init="snake_beta_init<APPROX_MODE>()",
    ),
)

_VARIANTS = [
    (case, formats_dest_acc, implied_math_format, vector_mode)
    for case in SFPI_TERNARY_CASES
    for formats_dest_acc, implied_math_format, vector_mode in product(
        case.formats,
        (ImpliedMathFormat.No, ImpliedMathFormat.Yes),
        (VectorMode.None_, VectorMode.R, VectorMode.C, VectorMode.RC),
    )
]


def _processed_face_mask(vector_mode):
    mask = torch.zeros(4 * _FACE_SIZE, dtype=torch.bool)
    for face in _PROCESSED_FACES[vector_mode]:
        mask[face * _FACE_SIZE : (face + 1) * _FACE_SIZE] = True
    return mask


def _generate_operand(data_format, seed, spec):
    torch.manual_seed(seed)
    operand, tile_count, _, _ = generate_stimuli(
        stimuli_format_A=data_format,
        input_dimensions_A=[32, 32],
        stimuli_format_B=data_format,
        input_dimensions_B=[32, 32],
        spec_A=spec,
        spec_B=spec,
    )
    return operand.flatten(), tile_count


@pytest.mark.quasar
@pytest.mark.parametrize(
    "case,formats_dest_acc,implied_math_format,vector_mode",
    _VARIANTS,
    ids=lambda value: repr(value) if isinstance(value, SfpiTernaryCase) else None,
)
def test_sfpi_compat_ternary_quasar(
    case, formats_dest_acc, implied_math_format, vector_mode
):
    header = _QSR_SFPU_DIR / f"ckernel_sfpu_{case.kernel}.h"
    if not header.is_file():
        pytest.skip(f"Quasar SFPI kernel has not landed yet: {header.name}")

    formats, dest_acc = formats_dest_acc
    spec_ab = StimuliSpec.uniform(low=-1.0, high=1.0)
    spec_c = (
        StimuliSpec.uniform(low=0.5, high=2.0)
        if case.mathop in (MathOperation.SfpuAddcdiv, MathOperation.SfpuSnakeBeta)
        else spec_ab
    )
    operand_a, tile_count = _generate_operand(formats.input_format, 42, spec_ab)
    operand_b, _ = _generate_operand(formats.input_format, 43, spec_ab)
    operand_c, _ = _generate_operand(formats.input_format, 44, spec_c)

    golden = get_golden_generator(TernarySFPUGolden)(
        case.mathop,
        operand_a,
        operand_b,
        operand_c,
        _SCALAR_BITS,
        formats.output_format,
    ).flatten()

    source = torch.cat((operand_a, operand_b, operand_c))
    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )
    configuration = TestConfig(
        _CPP_SOURCE,
        formats,
        templates=[
            MATH_OP(case.mathop),
            APPROX_MODE(ApproximationMode.No),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(),
            VECTOR_MODE(vector_mode),
            SFPU_TERNARY_SCALAR(_SCALAR_BITS),
            SFPI_COMPAT_TERNARY_KERNEL(case, formats.input_format),
        ],
        runtimes=[
            TILE_COUNT(tile_count * 3),
            NUM_FACES(4),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
        ],
        variant_stimuli=StimuliConfig(
            source,
            formats.input_format,
            torch.zeros_like(operand_a),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count * 3,
            tile_count_B=tile_count,
            tile_count_res=1,
            num_faces=4,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    result = torch.tensor(
        configuration.run().result, dtype=format_dict[formats.output_format]
    )
    mask = _processed_face_mask(vector_mode)
    assert passed_test(golden[mask], result[mask], formats.output_format)
