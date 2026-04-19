# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# AI-generated — run_id: 2026-04-02_trigonometry_quasar_e1448d06

from dataclasses import dataclass
from typing import List

import pytest
import torch
from helpers.format_config import DataFormat, FormatConfig
from helpers.golden_generators import UnarySFPUGolden, get_golden_generator
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    ImpliedMathFormat,
    MathOperation,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
    TemplateParameter,
)
from helpers.utils import passed_test

# Operation type constants matching the C++ SFPU_OP_TYPE dispatch:
# 0 = sine, 1 = cosine, 2 = acosh, 3 = asinh, 4 = atanh
TRIG_OP_SINE = 0
TRIG_OP_COSINE = 1
TRIG_OP_ACOSH = 2
TRIG_OP_ASINH = 3
TRIG_OP_ATANH = 4

# Map from operation type integer to MathOperation for golden reference
TRIG_OP_TO_MATH_OP = {
    TRIG_OP_SINE: MathOperation.Sin,
    TRIG_OP_COSINE: MathOperation.Cos,
    TRIG_OP_ACOSH: MathOperation.Acosh,
    TRIG_OP_ASINH: MathOperation.Asinh,
    TRIG_OP_ATANH: MathOperation.Atanh,
}

TRIG_OP_NAMES = {
    TRIG_OP_SINE: "Sine",
    TRIG_OP_COSINE: "Cosine",
    TRIG_OP_ACOSH: "Acosh",
    TRIG_OP_ASINH: "Asinh",
    TRIG_OP_ATANH: "Atanh",
}


@dataclass
class SFPU_OP_TYPE_PARAM(TemplateParameter):
    """Custom template parameter for trigonometry operation type dispatch."""

    op_type: int = 0

    def convert_to_cpp(self) -> str:
        return f"constexpr int SFPU_OP_TYPE = {self.op_type};"


def prepare_trig_inputs(
    src_A: torch.Tensor,
    op_type: int,
    input_format: DataFormat,
    output_format: DataFormat,
) -> torch.Tensor:
    """
    Prepare input tensor for trigonometry operations with safe value ranges.

    Each trig operation has different valid input domains:
    - sine: [-10, 10]
    - cosine: [-10, 10]
    - acosh: [1.0, 10.0]
    - asinh: [-10, 10]
    - atanh: [-0.95, 0.95]

    Args:
        src_A: Source tensor A (raw stimuli in [0, 1) range)
        op_type: Trig operation type constant
        input_format: Input data format
        output_format: Output data format

    Returns:
        Prepared tensor with safe values for the operation
    """
    torch_format = format_dict[input_format]

    if op_type == TRIG_OP_SINE:
        # sine: [-10, 10]
        min_val = -10.0
        max_val = 10.0
        src_A = min_val + src_A.to(torch.float32) * (max_val - min_val)
        src_A = src_A.to(torch_format)
    elif op_type == TRIG_OP_COSINE:
        # cosine: [-10, 10]
        min_val = -10.0
        max_val = 10.0
        src_A = min_val + src_A.to(torch.float32) * (max_val - min_val)
        src_A = src_A.to(torch_format)
    elif op_type == TRIG_OP_ACOSH:
        # acosh: domain is [1.0, inf), use [1.0, 10.0] for safe range
        # Use uniform distribution in [1.0, 10.0] to get good coverage
        min_val = 1.0
        max_val = 10.0
        src_A = min_val + src_A.to(torch.float32).abs() * (max_val - min_val)
        src_A = torch.clamp(src_A, min_val, max_val)
        src_A = src_A.to(torch_format)
    elif op_type == TRIG_OP_ASINH:
        # asinh: [-10, 10]
        min_val = -10.0
        max_val = 10.0
        src_A = min_val + src_A.to(torch.float32) * (max_val - min_val)
        src_A = src_A.to(torch_format)
    elif op_type == TRIG_OP_ATANH:
        # atanh: domain is (-1, 1), use [-0.95, 0.95] for safe range
        min_val = -0.95
        max_val = 0.95
        src_A = min_val + src_A.to(torch.float32) * (max_val - min_val)
        src_A = torch.clamp(src_A, min_val, max_val)
        src_A = src_A.to(torch_format)

    return src_A


def _is_invalid_quasar_combination(
    fmt: FormatConfig, dest_acc: DestAccumulation
) -> bool:
    """
    Check if format combination is invalid for Quasar.

    Args:
        fmt: Format configuration with input and output formats
        dest_acc: Destination accumulation mode

    Returns:
        True if the combination is invalid, False otherwise
    """
    in_fmt = fmt.input_format
    out_fmt = fmt.output_format

    # Quasar packer does not support non-Float32 to Float32 conversion when dest_acc=No
    if (
        in_fmt != DataFormat.Float32
        and out_fmt == DataFormat.Float32
        and dest_acc == DestAccumulation.No
    ):
        return True

    # Quasar SFPU with Float32 input and Float16 output requires dest_acc=Yes
    if (
        in_fmt == DataFormat.Float32
        and out_fmt == DataFormat.Float16
        and dest_acc == DestAccumulation.No
    ):
        return True

    return False


def generate_sfpu_trig_combinations(
    formats_list: List[FormatConfig],
):
    """
    Generate SFPU trigonometry test combinations.

    Args: Input-output format pairs

    Returns: List of (format, dest_acc, implied_math_format, input_dimensions, op_type) tuples
    """
    combinations = []

    for fmt in formats_list:
        in_fmt = fmt.input_format

        dest_acc_modes = (
            (DestAccumulation.Yes,)
            if in_fmt.is_32_bit()
            else (DestAccumulation.No, DestAccumulation.Yes)
        )
        for dest_acc in dest_acc_modes:
            # Skip invalid format combinations for Quasar
            if _is_invalid_quasar_combination(fmt, dest_acc):
                continue

            for implied_math_format in [ImpliedMathFormat.No, ImpliedMathFormat.Yes]:
                for input_dimensions in [[32, 32]]:
                    for op_type in [
                        TRIG_OP_SINE,
                        TRIG_OP_COSINE,
                        TRIG_OP_ACOSH,
                        TRIG_OP_ASINH,
                        TRIG_OP_ATANH,
                    ]:
                        combinations.append(
                            (
                                fmt,
                                dest_acc,
                                implied_math_format,
                                input_dimensions,
                                op_type,
                            )
                        )

    return combinations


SFPU_TRIG_FORMATS = input_output_formats(
    [
        DataFormat.Float16,
        DataFormat.Float32,
        DataFormat.Float16_b,
    ]
)


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_implied_math_input_dims_op=generate_sfpu_trig_combinations(
        SFPU_TRIG_FORMATS
    ),
)
def test_sfpu_trigonometry_quasar(formats_dest_acc_implied_math_input_dims_op):
    """
    Test trigonometric SFPU operations (sine, cosine, acosh, asinh, atanh) on Quasar architecture.

    Uses a compile-time SFPU_OP_TYPE integer for dispatch since SfpuType enum
    does not include trigonometry operations.
    """
    (formats, dest_acc, implied_math_format, input_dimensions, op_type) = (
        formats_dest_acc_implied_math_input_dims_op[0]
    )

    op_name = TRIG_OP_NAMES[op_type]
    math_op = TRIG_OP_TO_MATH_OP[op_type]

    # Set seed for reproducibility
    torch.manual_seed(42)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=False,
    )

    # Prepare inputs with operation-specific safe ranges
    src_A = prepare_trig_inputs(
        src_A, op_type, formats.input_format, formats.output_format
    )

    num_faces = 4

    generate_golden = get_golden_generator(UnarySFPUGolden)
    golden_tensor = generate_golden(
        math_op,
        src_A,
        formats.output_format,
        dest_acc,
        formats.input_format,
        input_dimensions,
    )

    # unpack_to_dest works only when format bit-width matches Dest mode
    unpack_to_dest = formats.input_format.is_32_bit() == (
        dest_acc == DestAccumulation.Yes
    )

    configuration = TestConfig(
        "sources/quasar/sfpu_trigonometry_quasar_test.cpp",
        formats,
        templates=[
            SFPU_OP_TYPE_PARAM(op_type=op_type),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    # Verify results match golden
    assert len(res_from_L1) == len(
        golden_tensor
    ), f"Result tensor and golden tensor are not of the same length for {op_name}"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), f"Assert against golden failed for {op_name}"
