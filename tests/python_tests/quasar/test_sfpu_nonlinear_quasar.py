# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

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
    INPUT_DIMENSIONS,
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.utils import passed_test


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


def generate_sfpu_nonlinear_combinations(
    formats_list: List[FormatConfig],
):
    """
    Generate SFPU nonlinear test combinations.

    Args: Input-output format pairs

    Returns: List of (format, dest_acc, implied_math_format, input_dimensions, mathop) tuples
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
                for input_dimensions in [[32, 32], [64, 64]]:
                    for mathop in [
                        MathOperation.Exp,
                        MathOperation.Relu,
                        MathOperation.Reciprocal,
                        MathOperation.Sqrt,
                        MathOperation.Tanh,
                    ]:
                        combinations.append(
                            (
                                fmt,
                                dest_acc,
                                implied_math_format,
                                input_dimensions,
                                mathop,
                            )
                        )

    return combinations


def prepare_inputs_for_operation(
    src_A: torch.Tensor,
    mathop: MathOperation,
    input_format: DataFormat,
    output_format: DataFormat = None,
) -> torch.Tensor:
    """
    Prepare input tensor for specific operation with safe value ranges.

    Args:
        src_A: Source tensor A
        mathop: Math operation to prepare inputs for
        input_format: Input data format
        output_format: Output data format, used for operations where safe ranges depend on the destination format
    Returns:
        Prepared tensor with safe values for the operation
    """
    torch_format = format_dict[input_format]

    if mathop == MathOperation.Exp:
        # Scale to range [-10, 10] for exp - avoids overflow while testing meaningful range
        # exp(-10) ≈ 0.000045, exp(10) ≈ 22026
        min_val = -10.0
        max_val = 10.0
        src_A = min_val + src_A.to(torch.float32) * (max_val - min_val)
        src_A = src_A.to(torch_format)
    elif mathop == MathOperation.Relu:
        # Scale to range including negative and positive values for ReLU testing
        finfo = torch.finfo(torch_format)
        min_val = finfo.min / 2  # Use half range to avoid extremes
        max_val = finfo.max / 2
        src_A = min_val + src_A.to(torch.float32) * (max_val - min_val)
        src_A = src_A.to(torch_format)
    elif mathop == MathOperation.Sqrt:
        # Scale to positive range using log-uniform distribution
        # sqrt only accepts non-negative inputs
        finfo = torch.finfo(torch_format)
        min_val = max(1e-6, finfo.tiny * 100)
        # Determine max value based on both input and output formats
        # When output is Float16, limit input range so sqrt results fit well in Float16
        # CRITICAL: The golden generator converts input to output format FIRST, then computes sqrt
        # So we must ensure input values fit in output format when converted
        if output_format:
            output_torch_format = format_dict[output_format]
            output_finfo = torch.finfo(output_torch_format)
            # For Float16 output, we need to ensure:
            # 1. Input fits in output format (doesn't overflow when converted)
            # 2. sqrt(input) fits in output format
            if output_torch_format in (torch.float16, torch.bfloat16):
                # CRITICAL: The golden generator converts input -> Float16 FIRST, then computes sqrt
                # So input must fit in Float16 when converted (this is the primary constraint!)
                # For Float16_b input -> Float16 output, Float16_b can represent much larger values
                # but they overflow Float16 when converted, causing sqrt(inf) = inf
                max_input_for_format = (
                    output_finfo.max
                )  # Input must fit in Float16 (~65504)
                # Also ensure sqrt(input) fits in Float16
                max_safe_sqrt = output_finfo.max * 0.95  # Leave 5% headroom
                max_input_for_sqrt = max_safe_sqrt**2  # Max input so sqrt fits (~3.9e9)
                # Use the more restrictive limit - input format max is usually larger, so this
                # will be limited by max_input_for_format (Float16.max)
                max_val = min(finfo.max, max_input_for_format, max_input_for_sqrt)
                # Additional safety: use 80% of Float16 max to avoid any edge cases
                # This ensures sqrt results are well within Float16 range
                max_val = min(max_val, output_finfo.max * 0.8)  # ~52400, sqrt ≈ 229
            else:
                max_val = finfo.max
        else:
            # No output format specified, use conservative limits for 16-bit input formats
            if torch_format in (torch.float16, torch.bfloat16):
                max_val = min(
                    finfo.max, 1e4
                )  # sqrt(1e4) = 100, safe for 16-bit formats
            else:
                max_val = finfo.max  # Float32 can handle larger values
        # Transform uniform [0,1) to log-uniform [min_val, max_val]
        log_min = torch.log(torch.tensor(min_val, dtype=torch.float32))
        log_max = torch.log(torch.tensor(float(max_val), dtype=torch.float32))
        src_A_float32 = torch.exp(
            log_min + src_A.to(torch.float32) * (log_max - log_min)
        )
        # Clamp to ensure values don't exceed max_val (handles any floating point precision issues)
        src_A_float32 = torch.clamp(src_A_float32, min_val, max_val)

        # Final safety check: ensure values fit in output format when converted
        # This is critical because the golden generator converts input -> output format first
        if output_format and output_format in (
            DataFormat.Float16,
            DataFormat.Float16_b,
        ):
            output_torch_format = format_dict[output_format]
            output_finfo = torch.finfo(output_torch_format)
            # Convert to output format to check for overflow
            src_A_converted = src_A_float32.to(output_torch_format)
            if torch.any(torch.isinf(src_A_converted)):
                # If any values overflow, clamp more aggressively
                # Use 80% of Float16 max to leave plenty of headroom
                max_safe_input = output_finfo.max * 0.8
                src_A_float32 = torch.clamp(src_A_float32, min_val, max_safe_input)

        src_A = src_A_float32.to(torch_format)

        # Additional check: after converting to input format, verify values still fit in output format
        # This handles cases where input format (e.g., Float16_b) can represent larger values
        # than output format (e.g., Float16)
        if output_format and output_format in (
            DataFormat.Float16,
            DataFormat.Float16_b,
        ):
            output_torch_format = format_dict[output_format]
            output_finfo = torch.finfo(output_torch_format)
            # Convert input format -> output format to check for overflow
            src_A_converted = src_A.to(output_torch_format)
            if torch.any(torch.isinf(src_A_converted)):
                # If overflow occurs, clamp input values to output format max
                max_safe_input = output_finfo.max * 0.75  # Very conservative
                # Convert back to float32, clamp, then convert to input format
                src_A_float32 = src_A.to(torch.float32)
                src_A_float32 = torch.clamp(src_A_float32, min_val, max_safe_input)
                src_A = src_A_float32.to(torch_format)
    elif mathop == MathOperation.Reciprocal:
        # Scale to range avoiding zero to prevent division by zero
        # Reciprocal: 1/x, so we need to avoid x = 0
        finfo = torch.finfo(torch_format)
        # Use a range that avoids very small values near zero
        min_val = max(1e-6, finfo.tiny * 100)
        max_val = finfo.max / 2  # Avoid very large values that might cause underflow
        # Use log-uniform distribution to test across orders of magnitude
        log_min = torch.log(torch.tensor(min_val, dtype=torch.float32))
        log_max = torch.log(torch.tensor(float(max_val), dtype=torch.float32))
        src_A_float32 = torch.exp(
            log_min + src_A.to(torch.float32) * (log_max - log_min)
        )
        # Ensure no values are too close to zero
        src_A_float32 = torch.where(
            torch.abs(src_A_float32) < min_val,
            torch.sign(src_A_float32) * min_val,
            src_A_float32,
        )
        src_A = src_A_float32.to(torch_format)
    elif mathop == MathOperation.Tanh:
        # Scale to range [-10, 10] for tanh - covers meaningful range without saturation
        min_val = -10.0
        max_val = 10.0
        src_A = min_val + src_A.to(torch.float32) * (max_val - min_val)
        src_A = src_A.to(torch_format)
    # else: keep src_A as-is for other operations

    return src_A


SFPU_NONLINEAR_FORMATS = input_output_formats(
    [
        DataFormat.Float16,
        DataFormat.Float32,
        DataFormat.Float16_b,
    ]
)


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_implied_math_input_dims_mathop=generate_sfpu_nonlinear_combinations(
        SFPU_NONLINEAR_FORMATS
    ),
)
def test_sfpu_nonlinear_quasar(formats_dest_acc_implied_math_input_dims_mathop):
    """
    Test nonlinear SFPU operations (exp, relu, reciprocal, sqrt, tanh) on Quasar architecture.

    This test parameterizes over multiple operations to avoid code duplication.
    """
    (formats, dest_acc, implied_math_format, input_dimensions, mathop) = (
        formats_dest_acc_implied_math_input_dims_mathop[0]
    )

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=False,
    )

    # Prepare inputs with operation-specific ranges
    src_A = prepare_inputs_for_operation(
        src_A, mathop, formats.input_format, formats.output_format
    )

    num_faces = 4

    generate_golden = get_golden_generator(UnarySFPUGolden)
    golden_tensor = generate_golden(
        mathop,
        src_A,
        formats.output_format,
        dest_acc,
        formats.input_format,
        input_dimensions,
    )

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )
    configuration = TestConfig(
        "sources/quasar/sfpu_nonlinear_quasar_test.cpp",
        formats,
        templates=[
            INPUT_DIMENSIONS(input_dimensions, input_dimensions),
            MATH_OP(mathop=mathop),
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

    res_from_L1 = configuration.run()

    # Verify results match golden
    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
