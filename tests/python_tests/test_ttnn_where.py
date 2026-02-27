# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, MathOperation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DISABLE_SRC_ZERO_FLAG,
    MATH_OP,
)


def generate_golden(operand1, true_value, false_value):
    # operand1, true_value, and false_value are 1D tensors of floats
    mask = operand1.view(32, 32) != 0
    return torch.where(
        mask, true_value.view(32, 32), false_value.view(32, 32)
    ).flatten()


# Helper check function
def torch_equal_nan(a, b):
    return torch.all((a == b) | (torch.isnan(a) & torch.isnan(b)))


@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float32,
            DataFormat.Int32,
        ],
        same=True,
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=MathOperation.TTNNWhere,
    test_case=["mixed", "all_ones", "all_zeros"],
)
def test_ttnn_where(formats, dest_acc, mathop, test_case, workers_tensix_coordinates):

    if (
        formats.input == DataFormat.Float32 and formats.output == DataFormat.Float32
    ) and dest_acc == DestAccumulation.No:
        pytest.skip("DataFormat.Float32 not supported with DestAccumulation.No")

    if (
        formats.input == DataFormat.Float16_b and formats.output == DataFormat.Float16_b
    ) and dest_acc == DestAccumulation.Yes:
        pytest.skip("DataFormat.Float16_b not supported with DestAccumulation.Yes")

    input_dimensions = [32, 32]  # Single tile dimensions
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=False,
    )

    src_C, tile_cnt_C, _, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        sfpu=False,
    )

    # Modify the condition tensor based on test case
    if test_case == "all_ones":
        src_A = torch.ones_like(src_A)
    elif test_case == "all_zeros":
        src_A = torch.zeros_like(src_A)
    # For "mixed" case, use the generated stimuli as-is

    golden = generate_golden(src_A, src_B, src_C)

    configuration = TestConfig(
        "sources/ttnn_where_test.cpp",
        formats,
        templates=[MATH_OP(mathop), DISABLE_SRC_ZERO_FLAG(True)],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A.flatten(),
            formats.input_format,
            src_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            buffer_C=src_C.flatten(),
            stimuli_C_format=formats.input_format,
            tile_count_C=tile_cnt_C,
        ),
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res_from_L1 = configuration.run(workers_tensix_coordinates).result

    res_from_L1 = res_from_L1[:1024]
    assert len(res_from_L1) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"

    golden_tensor = torch.tensor(
        golden,
        dtype=(
            format_dict[formats.output_format]
            if formats.output_format in [DataFormat.Float16_b, DataFormat.Float32]
            else torch.bfloat16
        ),
    )
    res_tensor = torch.tensor(
        res_from_L1,
        dtype=(
            format_dict[formats.output_format]
            if formats.output_format in [DataFormat.Float16_b, DataFormat.Float32]
            else torch.bfloat16
        ),
    )

    assert torch_equal_nan(golden_tensor, res_tensor), "Assert against golden failed"


# MCW test with dynamic format sweeping like main test
# Use same input/output format - no mixing
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float32,
            DataFormat.Int32,
        ],
        same=True,
    ),
    dest_acc=[DestAccumulation.No, DestAccumulation.Yes],
    mathop=MathOperation.TTNNWhere,
    height=[32],
    width=[32],
)
def test_ttnn_where_mcw(
    formats, dest_acc, mathop, height, width, workers_tensix_coordinates
):
    # Generate dtype dynamically based on current input format

    if (
        formats.input == DataFormat.Float32 and formats.output == DataFormat.Float32
    ) and dest_acc == DestAccumulation.No:
        pytest.skip("DataFormat.Float32 not supported with DestAccumulation.No")

    if (
        formats.input == DataFormat.Float16_b and formats.output == DataFormat.Float16_b
    ) and dest_acc == DestAccumulation.Yes:
        pytest.skip("DataFormat.Float16_b not supported with DestAccumulation.Yes")

    # Create alternating pattern for condition (0, 1, 0, 1, ...)
    pattern = torch.arange(height * width) % 2
    C = pattern.view(height, width).to(format_dict[formats.input_format])

    # Set specific values for true and false tensors
    T = torch.ones(height, width, dtype=format_dict[formats.input_format]) * 2
    F = torch.ones(height, width, dtype=format_dict[formats.input_format]) * 11

    golden = generate_golden(C, T, F)

    configuration = TestConfig(
        "sources/ttnn_where_test.cpp",
        formats,
        templates=[MATH_OP(mathop), DISABLE_SRC_ZERO_FLAG(True)],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            C.flatten(),
            formats.input_format,
            T.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
            buffer_C=F.flatten(),
            stimuli_C_format=formats.input_format,
            tile_count_C=1,
        ),
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res_from_L1 = configuration.run(workers_tensix_coordinates).result

    res_from_L1 = res_from_L1[:1024]

    golden_tensor = torch.tensor(
        golden,
        dtype=(
            format_dict[formats.output_format]
            if formats.output_format in [DataFormat.Float16_b, DataFormat.Float32]
            else torch.bfloat16
        ),
    )

    golden_tensor = golden_tensor.flatten()[:1024]  # Ensure it matches the result size

    res_tensor = torch.tensor(
        res_from_L1,
        dtype=(
            format_dict[formats.output_format]
            if formats.output_format in [DataFormat.Float16_b, DataFormat.Float32]
            else torch.bfloat16
        ),
    )

    assert len(res_tensor) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"
    assert torch_equal_nan(golden_tensor, res_tensor), "Assert against golden failed"
