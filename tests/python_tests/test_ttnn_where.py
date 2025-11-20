# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from helpers.device import (
    collect_results,
    write_stimuli_to_l1,
)
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, MathOperation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import run_test


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
    test_name="ttnn_where_test",
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
def test_ttnn_where(test_name, formats, dest_acc, mathop, test_case):

    if (
        formats.input == DataFormat.Float32 and formats.output == DataFormat.Float32
    ) and dest_acc == DestAccumulation.No:
        pytest.skip("DataFormat.Float32 not supported with DestAccumulation.No")

    if (
        formats.input == DataFormat.Float16_b and formats.output == DataFormat.Float16_b
    ) and dest_acc == DestAccumulation.Yes:
        pytest.skip("DataFormat.Float16_b not supported with DestAccumulation.Yes")

    input_dimensions = [32, 32]  # Single tile dimensions

    src_A, _, tile_cnt_A = generate_stimuli(
        formats.input_format,
        formats.input_format,
        input_dimensions=input_dimensions,
        sfpu=False,
    )
    src_B, _, tile_cnt_B = generate_stimuli(
        formats.input_format,
        formats.input_format,
        input_dimensions=input_dimensions,
        sfpu=False,
    )
    src_C, _, tile_cnt_C = generate_stimuli(
        formats.input_format,
        formats.input_format,
        input_dimensions=input_dimensions,
        sfpu=False,
    )

    # Modify the condition tensor based on test case
    if test_case == "all_ones":
        src_A = torch.ones_like(src_A)
    elif test_case == "all_zeros":
        src_A = torch.zeros_like(src_A)
    # For "mixed" case, use the generated stimuli as-is

    location = "0,0"

    golden = generate_golden(src_A, src_B, src_C)

    # Create test config for storing buffer addresses
    buffer_config = {}

    # Write all three inputs using the enhanced helper function
    result_buffer_address = write_stimuli_to_l1(
        test_config=buffer_config,
        buffer_A=src_A.flatten(),
        buffer_B=src_B.flatten(),
        stimuli_A_format=formats.input_format,
        stimuli_B_format=formats.input_format,
        tile_count_A=tile_cnt_A,
        tile_count_B=tile_cnt_B,
        location=location,
        buffer_C=src_C.flatten(),
        stimuli_C_format=formats.input_format,
        tile_count_C=tile_cnt_C,
    )

    unpack_to_dest = formats.input_format.is_32_bit()

    test_config = {
        "formats": formats,
        "testname": test_name,
        "dest_acc": dest_acc,
        "mathop": mathop,
        "unpack_to_dest": unpack_to_dest,
        "buffer_A_address": buffer_config["buffer_A_address"],
        "buffer_B_address": buffer_config["buffer_B_address"],
        "buffer_C_address": buffer_config["buffer_C_address"],
        "result_buffer_address": buffer_config["result_buffer_address"],
        "tile_cnt_A": tile_cnt_A,
        "tile_cnt_B": tile_cnt_B,
        "tile_cnt_C": tile_cnt_C,
    }

    run_test(test_config)

    res_from_L1 = collect_results(
        formats, tile_count=tile_cnt_A, address=result_buffer_address
    )
    res_from_L1 = res_from_L1[:1024]
    assert len(res_from_L1) == len(golden)

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

    assert torch_equal_nan(golden_tensor, res_tensor)


# MCW test with dynamic format sweeping like main test
# Use same input/output format - no mixing
@parametrize(
    test_name="ttnn_where_test",
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
def test_ttnn_where_mcw(test_name, formats, dest_acc, mathop, height, width):
    # Generate dtype dynamically based on current input format

    if (
        formats.input == DataFormat.Float32 and formats.output == DataFormat.Float32
    ) and dest_acc == DestAccumulation.No:
        pytest.skip("DataFormat.Float32 not supported with DestAccumulation.No")

    if (
        formats.input == DataFormat.Float16_b and formats.output == DataFormat.Float16_b
    ) and dest_acc == DestAccumulation.Yes:
        pytest.skip("DataFormat.Float16_b not supported with DestAccumulation.Yes")

    # Generate stimuli using the standard helper function
    input_dimensions = [height, width]

    C, _, tile_cnt_C = generate_stimuli(
        formats.input_format,
        formats.input_format,
        input_dimensions=input_dimensions,
        sfpu=False,
    )
    T, _, tile_cnt_T = generate_stimuli(
        formats.input_format,
        formats.input_format,
        input_dimensions=input_dimensions,
        sfpu=False,
    )
    F, _, tile_cnt_F = generate_stimuli(
        formats.input_format,
        formats.input_format,
        input_dimensions=input_dimensions,
        sfpu=False,
    )

    # Create alternating pattern for condition (0, 1, 0, 1, ...)
    pattern = torch.arange(height * width) % 2
    C = pattern.view(height, width).to(format_dict[formats.input_format])

    # Set specific values for true and false tensors
    T = torch.ones(height, width, dtype=format_dict[formats.input_format]) * 2
    F = torch.ones(height, width, dtype=format_dict[formats.input_format]) * 11

    location = "0,0"

    golden = generate_golden(C, T, F)

    # Create test config for storing buffer addresses
    buffer_config = {}

    result_buffer_address = write_stimuli_to_l1(
        test_config=buffer_config,
        buffer_A=C.flatten(),
        buffer_B=T.flatten(),
        stimuli_A_format=formats.input_format,
        stimuli_B_format=formats.input_format,
        tile_count_A=tile_cnt_C,
        tile_count_B=tile_cnt_T,
        location=location,
        buffer_C=F.flatten(),
        stimuli_C_format=formats.input_format,
        tile_count_C=tile_cnt_F,
    )

    unpack_to_dest = formats.input_format.is_32_bit()

    test_config = {
        "formats": formats,
        "testname": test_name,
        "dest_acc": dest_acc,
        "unpack_to_dest": unpack_to_dest,
        "mathop": mathop,
        "buffer_A_address": buffer_config["buffer_A_address"],
        "buffer_B_address": buffer_config["buffer_B_address"],
        "buffer_C_address": buffer_config["buffer_C_address"],
        "result_buffer_address": buffer_config["result_buffer_address"],
        "tile_cnt_A": tile_cnt_C,
        "tile_cnt_B": tile_cnt_T,
        "tile_cnt_C": tile_cnt_F,
    }

    run_test(test_config)

    res_from_L1 = collect_results(
        formats, tile_count=tile_cnt_C, address=result_buffer_address
    )
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

    assert len(res_tensor) == len(golden_tensor)
    assert torch_equal_nan(golden_tensor, res_tensor)
