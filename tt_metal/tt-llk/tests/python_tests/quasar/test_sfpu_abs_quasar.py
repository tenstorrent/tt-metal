# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# AI-generated — run_id: 2026-04-08_abs_quasar_2f52d870

from typing import List

import pytest
import torch
from helpers.format_config import (
    MXFP8_E4M3_MAX_NORMAL,
    MXFP8_E5M2_MAX_NORMAL,
    DataFormat,
    FormatConfig,
)
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
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.utils import passed_test


def _prepare_int8_abs_inputs(src_A: torch.Tensor, src_B: torch.Tensor) -> torch.Tensor:
    """
    Int8 stimulus for abs. Range is clamped to [-127, 127] to avoid the
    unrepresentable -128 (Dest int8 is sign+magnitude — -128 has no sign+mag
    encoding and is saturated to -127 at unpack, breaking bit-exact compare).
    Uses src_A magnitude and src_B sign to reuse the shared stimulus seeding.
    """
    # Map whatever dtype generate_stimuli produced into a usable integer range.
    # src_A may be float or int depending on prior conversions; coerce to int32
    # for arithmetic, then clamp + cast to int8.
    magnitudes = src_A.to(torch.int32).abs() % 128  # [0, 127]
    signs = torch.where(src_B.to(torch.int32) % 2 == 0, 1, -1)
    values = (signs * magnitudes).clamp(-127, 127).to(torch.int8)
    return values


def _prepare_uint8_abs_inputs(src_A: torch.Tensor) -> torch.Tensor:
    """
    UInt8 stimulus for abs. Abs is identity for unsigned values, so the full
    0-255 range is valid; just ensure the tensor is the right dtype.
    """
    return src_A.to(torch.int32).abs().clamp(0, 255).to(torch.uint8)


def _prepare_int32_abs_inputs(src_A: torch.Tensor, src_B: torch.Tensor) -> torch.Tensor:
    """
    Int32 stimulus for abs. Dest int32 is sign+magnitude, so 0x80000000 is
    unrepresentable (saturated to 0x7FFFFFFF at unpack). Clamp to
    [-(2^31 - 1), 2^31 - 1] so the torch golden matches HW bit-exact.
    """
    int32_max = 2**31 - 1
    magnitudes = src_A.to(torch.int64).abs() % (int32_max + 1)  # [0, 2^31-1]
    signs = torch.where(src_B.to(torch.int64) % 2 == 0, 1, -1)
    values = (signs * magnitudes).clamp(-int32_max, int32_max).to(torch.int32)
    return values


def prepare_abs_inputs(
    src_A: torch.Tensor,
    src_B: torch.Tensor,
    input_format: DataFormat,
    output_format: DataFormat,
) -> torch.Tensor:
    """
    Prepare input tensor for absolute value operation with safe value ranges.

    Generates a mix of positive and negative values across the representable
    range. Abs does not change magnitude, so the only constraint is that values
    fit in the input/output format.

    Args:
        src_A: Source tensor A (used for magnitude distribution)
        src_B: Source tensor B (used for sign distribution)
        input_format: Input data format
        output_format: Output data format

    Returns:
        Prepared tensor with safe values for abs
    """
    if input_format == DataFormat.Int8:
        return _prepare_int8_abs_inputs(src_A, src_B)
    if input_format == DataFormat.UInt8:
        return _prepare_uint8_abs_inputs(src_A)
    if input_format == DataFormat.Int32:
        return _prepare_int32_abs_inputs(src_A, src_B)

    input_torch_format = format_dict[input_format]
    output_torch_format = format_dict[output_format]
    input_finfo = torch.finfo(input_torch_format)
    output_finfo = torch.finfo(output_torch_format)

    def _mx_elem_max(df: DataFormat) -> float:
        if df == DataFormat.MxFp8R:
            return MXFP8_E5M2_MAX_NORMAL
        if df == DataFormat.MxFp8P:
            return MXFP8_E4M3_MAX_NORMAL
        raise ValueError(f"not an MX format: {df}")

    # For abs, output magnitude equals input magnitude, so we need values that
    # fit in BOTH input and output formats. MX formats have their own element
    # max (not queryable via torch.finfo on bfloat16).
    cap_from_input = (
        _mx_elem_max(input_format) if input_format.is_mx_format() else input_finfo.max
    )
    cap_from_output = (
        _mx_elem_max(output_format)
        if output_format.is_mx_format()
        else output_finfo.max
    )
    max_safe_value = min(cap_from_input, cap_from_output) * 0.9

    # Special handling for bfloat16: limit to reasonable bounds to avoid
    # precision issues at extreme values
    if input_torch_format == torch.bfloat16:
        max_safe_value = min(max_safe_value, 1e4)
    else:
        max_safe_value = min(max_safe_value, input_finfo.max * 0.9)

    min_magnitude = max(1e-6, input_finfo.tiny * 100)  # Avoid denormals

    # Ensure src_A and src_B don't contain inf/nan before normalization
    src_A_float = src_A.to(torch.float32)
    src_B_float = src_B.to(torch.float32)

    # Normalize src_A to [0, 1] range for log-uniform distribution
    src_A_min = src_A_float.min()
    src_A_max = src_A_float.max()
    src_A_normalized = (
        (src_A_float - src_A_min) / (src_A_max - src_A_min)
        if src_A_max > src_A_min
        else torch.zeros_like(src_A_float)
    )

    # Use log-uniform distribution for magnitudes to test across orders of magnitude
    log_min = torch.log(torch.tensor(min_magnitude, dtype=torch.float32))
    log_max = torch.log(torch.tensor(max_safe_value, dtype=torch.float32))
    magnitudes = torch.exp(log_min + src_A_normalized * (log_max - log_min))

    # Randomly assign signs to get both positive and negative values
    src_B_min = src_B_float.min()
    src_B_max = src_B_float.max()
    src_B_normalized = (
        (src_B_float - src_B_min) / (src_B_max - src_B_min)
        if src_B_max > src_B_min
        else torch.zeros_like(src_B_float)
    )
    signs = torch.where(src_B_normalized < 0.5, -1.0, 1.0)

    # Apply signs and clamp to safe range BEFORE converting to input format
    src_A_values = signs * magnitudes
    src_A_values = torch.clamp(src_A_values, -max_safe_value, max_safe_value)
    result = src_A_values.to(input_torch_format)

    return result


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

    # Integer and float formats cannot be mixed in input/output
    if in_fmt.is_integer() != out_fmt.is_integer():
        return True

    # Cross-width integer conversions (e.g. Int32 -> Int8, UInt8 -> Int32) aren't
    # abs-testable: 8->32 requires dest_acc=Yes (invalid for 8-bit SFPU input
    # path, rejected by format inference) and 32->8 passes through the packer's
    # Int32 conversion path, which saturates rather than preserving the abs
    # value. Restrict to same-width integer pairs (Int8<->UInt8, Int32<->Int32).
    if in_fmt.is_integer() and out_fmt.is_integer() and in_fmt.size != out_fmt.size:
        return True

    # 8-bit integer data lives in 16-bit Dest (1 datum per row); dest_acc=Yes
    # uses 32-bit Dest mode intended for Float32/Int32 accumulation and is not
    # meaningful for 8-bit integer data paths.
    if (
        in_fmt in (DataFormat.Int8, DataFormat.UInt8)
        and dest_acc == DestAccumulation.Yes
    ):
        return True

    return False


def generate_sfpu_abs_combinations(
    formats_list: List[FormatConfig],
):
    """
    Generate SFPU abs test combinations.

    Args: Input-output format pairs

    Returns: List of (format, dest_acc, implied_math_format, input_dimensions) tuples
    """
    combinations = []

    for fmt in formats_list:
        in_fmt = fmt.input_format

        if in_fmt.is_32_bit():
            dest_acc_modes = (DestAccumulation.Yes,)
        elif in_fmt.is_mx_format():
            # MX tiles unpack to Float16_b in Dest; 32-bit Dest accumulation is
            # not the intended path.
            dest_acc_modes = (DestAccumulation.No,)
        else:
            dest_acc_modes = (DestAccumulation.No, DestAccumulation.Yes)

        for dest_acc in dest_acc_modes:
            # Skip invalid format combinations for Quasar
            if _is_invalid_quasar_combination(fmt, dest_acc):
                continue

            for implied_math_format in [ImpliedMathFormat.No, ImpliedMathFormat.Yes]:
                # MX formats require implied_math_format=Yes
                if (
                    in_fmt.is_mx_format()
                    and implied_math_format == ImpliedMathFormat.No
                ):
                    continue

                for input_dimensions in [[32, 32], [64, 64], [32, 64]]:
                    combinations.append(
                        (fmt, dest_acc, implied_math_format, input_dimensions)
                    )

    return combinations


SFPU_ABS_FORMATS = input_output_formats(
    [
        DataFormat.Float16,
        DataFormat.Float32,
        DataFormat.Float16_b,
        DataFormat.Int8,
        DataFormat.UInt8,
        DataFormat.Int32,
        DataFormat.MxFp8R,
        DataFormat.MxFp8P,
    ]
)


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_implied_math_input_dims=generate_sfpu_abs_combinations(
        SFPU_ABS_FORMATS
    ),
)
def test_sfpu_abs_quasar(formats_dest_acc_implied_math_input_dims):
    """
    Test absolute value operation on Quasar architecture.

    Uses Python's abs() as the golden reference. Abs is an exact operation
    (sign bit clear for float), so results should match bitwise.
    """
    (formats, dest_acc, implied_math_format, input_dimensions) = (
        formats_dest_acc_implied_math_input_dims[0]
    )

    # Set seed for reproducibility
    torch.manual_seed(42)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=True,
    )

    # Prepare inputs with a mix of positive and negative values for abs testing
    src_A = prepare_abs_inputs(
        src_A, src_B, formats.input_format, formats.output_format
    )

    num_faces = 4

    if formats.input_format.is_integer():
        # UnarySFPUGolden has no integer branch (it casts through Float16_b for
        # non-MX, non-Float16 inputs at dest_acc=No), which would drop integer
        # semantics. Abs on the clamped integer stimulus is an exact
        # element-wise op, so compute the golden directly in the input dtype.
        golden_tensor = (
            torch.abs(src_A).flatten().to(format_dict[formats.output_format])
        )
    else:
        generate_golden = get_golden_generator(UnarySFPUGolden)
        golden_tensor = generate_golden(
            MathOperation.Abs,
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
        "sources/quasar/sfpu_abs_quasar_test.cpp",
        formats,
        templates=[
            MATH_OP(mathop=MathOperation.Abs),
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
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
