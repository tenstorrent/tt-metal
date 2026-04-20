# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Isolated SFPU add (binary): UNPACK2 (UNP_S) x2 -> SrcS -> SFPU -> PACK1 -> L1.
No MATH kernel. Two operands unpacked to SrcS slices 0 and 1, added by SFPU,
result packed from SrcS slice 2 to L1.
"""

from typing import List

import pytest
import torch
from helpers.format_config import (
    MXFP8_E4M3_MAX_NORMAL,
    MXFP8_E4M3_MIN_MAGNITUDE,
    MXFP8_E5M2_MAX_NORMAL,
    MXFP8_E5M2_MIN_MAGNITUDE,
    DataFormat,
    FormatConfig,
)
from helpers.golden_generators import BinarySFPUGolden, get_golden_generator
from helpers.llk_params import (
    DestAccumulation,
    ImpliedMathFormat,
    MathOperation,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_INDEX,
    IMPLIED_MATH_FORMAT,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
)
from helpers.utils import passed_test


def prepare_add_inputs(
    src_A: torch.Tensor,
    src_B: torch.Tensor,
    input_format: DataFormat,
    output_format: DataFormat,
):
    """
    Prepare input tensors for add operation with safe value ranges.

    Clamps values so that a + b fits in both the input and output formats.

    Args:
        src_A: Source tensor A
        src_B: Source tensor B
        input_format: Input data format
        output_format: Output data format

    Returns:
        Tuple of prepared (src_A, src_B) tensors
    """
    input_torch_format = format_dict[input_format]
    output_torch_format = format_dict[output_format]
    input_finfo = torch.finfo(input_torch_format)
    output_finfo = torch.finfo(output_torch_format)

    def mx_elem_max(df: DataFormat) -> float:
        if df == DataFormat.MxFp8R:
            return MXFP8_E5M2_MAX_NORMAL
        if df == DataFormat.MxFp8P:
            return MXFP8_E4M3_MAX_NORMAL
        raise ValueError(f"mx_elem_max: not an MX format: {df}")

    # Safety factor applied to format maxima when clamping add operands.
    # Ensures |a| + |b| stays within representable range with headroom for
    # rounding/quantization (two operands each capped at 45% of max -> sum <= 90% of max).
    ADD_RANGE_SAFETY_FACTOR = 0.45

    if output_format.is_mx_format():
        cap_from_output = mx_elem_max(output_format) * ADD_RANGE_SAFETY_FACTOR
    else:
        cap_from_output = output_finfo.max * ADD_RANGE_SAFETY_FACTOR

    if input_format.is_mx_format():
        cap_from_input = mx_elem_max(input_format) * ADD_RANGE_SAFETY_FACTOR
    else:
        cap_from_input = input_finfo.max * ADD_RANGE_SAFETY_FACTOR

    max_safe_value = min(cap_from_output, cap_from_input)

    # bfloat16: limit magnitude for reasonable precision
    if input_torch_format == torch.bfloat16 and not input_format.is_mx_format():
        max_safe_value = min(max_safe_value, 1e4)

    # Use format-appropriate minimum magnitude.
    # MX formats map to torch.bfloat16 in format_dict, but actual FP8 element
    # types have much larger minimums than bfloat16.tiny.  Using bfloat16.tiny
    # produces values far below FP8 representable range, causing quantization
    # mismatches between golden and hardware.
    if input_format.is_mx_format():
        if input_format == DataFormat.MxFp8P:
            min_magnitude = MXFP8_E4M3_MIN_MAGNITUDE
        else:
            min_magnitude = MXFP8_E5M2_MIN_MAGNITUDE
    else:
        min_magnitude = max(1e-6, input_finfo.tiny * 100)

    # Also respect output format minimum if output is MX
    if output_format.is_mx_format():
        if output_format == DataFormat.MxFp8P:
            min_magnitude = max(min_magnitude, MXFP8_E4M3_MIN_MAGNITUDE)
        else:
            min_magnitude = max(min_magnitude, MXFP8_E5M2_MIN_MAGNITUDE)

    def clamp_tensor(src: torch.Tensor) -> torch.Tensor:
        src_float = src.to(torch.float32)
        src_min = src_float.min()
        src_max = src_float.max()
        normalized = (
            (src_float - src_min) / (src_max - src_min)
            if src_max > src_min
            else torch.zeros_like(src_float)
        )
        log_min = torch.log(torch.tensor(min_magnitude, dtype=torch.float32))
        log_max = torch.log(torch.tensor(max_safe_value, dtype=torch.float32))
        magnitudes = torch.exp(log_min + normalized * (log_max - log_min))
        # Alternate signs based on element index for coverage
        signs = torch.where(
            torch.arange(src.numel()) % 3 == 0,
            torch.tensor(-1.0),
            torch.tensor(1.0),
        )
        values = signs * magnitudes
        values = torch.clamp(values, -max_safe_value, max_safe_value)
        return values.to(input_torch_format)

    return clamp_tensor(src_A), clamp_tensor(src_B)


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


def generate_sfpu_add_combinations(
    formats_list: List[FormatConfig],
):
    """
    Generate SFPU add test combinations.

    Args: Input-output format pairs

    Returns: List of (format, dest_acc, implied_math_format, input_dimensions) tuples
    """
    combinations = []

    for fmt in formats_list:
        in_fmt = fmt.input_format

        if in_fmt.is_32_bit():
            dest_acc_modes = (DestAccumulation.Yes,)
        elif in_fmt.is_mx_format():
            dest_acc_modes = (DestAccumulation.No,)
        else:
            dest_acc_modes = (DestAccumulation.No, DestAccumulation.Yes)

        for dest_acc in dest_acc_modes:
            if _is_invalid_quasar_combination(fmt, dest_acc):
                continue

            for implied_math_format in [ImpliedMathFormat.No, ImpliedMathFormat.Yes]:
                for input_dimensions in [[32, 32], [64, 64], [32, 64]]:
                    combinations.append(
                        (fmt, dest_acc, implied_math_format, input_dimensions)
                    )

    return combinations


SFPU_ADD_FORMATS = input_output_formats(
    [
        DataFormat.MxFp8R,
        DataFormat.MxFp8P,
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Float32,
    ]
)


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_implied_math_input_dims=generate_sfpu_add_combinations(
        SFPU_ADD_FORMATS
    ),
)
def test_isolate_sfpu_add_quasar(formats_dest_acc_implied_math_input_dims):
    """
    Test isolated SFPU add (binary): UNPACK2 (UNP_S) x2 -> SrcS -> SFPU -> PACK1 -> L1.
    No MATH kernel (stub only). Two input operands unpacked to SrcS, added, packed.
    """
    (formats, dest_acc, implied_math_format, input_dimensions) = (
        formats_dest_acc_implied_math_input_dims[0]
    )

    torch.manual_seed(42)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=True,
    )

    src_A, src_B = prepare_add_inputs(
        src_A, src_B, formats.input_format, formats.output_format
    )

    num_faces = 4

    # Golden: use BinarySFPUGolden so we can swap ops for future binary kernels.
    # SrcS path is untilized, so skip_tilize=True. Concatenate full tensors:
    # [all A tiles | all B tiles], then index by tile count offset.
    generate_golden = get_golden_generator(BinarySFPUGolden)
    golden_tensor = generate_golden(
        MathOperation.SfpuElwadd,
        torch.cat([src_A, src_B]),
        0,  # src1_idx: first tile of A
        tile_cnt_A,  # src2_idx: first tile of B
        0,  # dst_idx: write result starting at tile 0
        tile_cnt_A * 32,  # num_iterations: 32 rows per tile
        [input_dimensions[0] * 2, input_dimensions[1]],
        formats.output_format,
        skip_tilize=True,
        input_format=formats.input_format,
    )[
        : src_A.numel()
    ]  # Extract only the result region (A's tiles)

    configuration = TestConfig(
        "sources/quasar/isolate_sfpu_add_quasar_test.cpp",
        formats,
        templates=[
            IMPLIED_MATH_FORMAT(implied_math_format),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(),
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
        unpack_to_srcs=True,
        dest_acc=dest_acc,
        disable_format_inference=formats.input_format.is_mx_format(),
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(golden_tensor)
    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)
    assert passed_test(golden_tensor, res_tensor, formats.output_format)
