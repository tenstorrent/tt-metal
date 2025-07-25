# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from ttexalens.tt_exalens_lib import (
    write_to_device,
)

from helpers.device import (
    collect_results,
    wait_for_tensix_operations_finished,
)
from helpers.format_arg_mapping import (
    DestAccumulation,
    MathOperation,
    format_dict,
)
from helpers.format_config import DataFormat
from helpers.pack import pack_bfp16, pack_fp32
from helpers.param_config import (
    input_output_formats,
    parametrize,
)
from helpers.test_config import run_test


# Helper function
def extend_tensor(condition, length=1024, dtype=torch.float32):
    condition_extended = torch.zeros(length, dtype=dtype)
    condition_extended[: condition.shape[0]] = condition
    return condition_extended.flatten()


def generate_golden(operand1, true_value, false_value):
    # operand1, true_value, and false_value are 1D tensors of floats
    mask = operand1.view(32, 32) != 0
    return torch.where(
        mask, true_value.view(32, 32), false_value.view(32, 32)
    ).flatten()


# Helper check function
def torch_equal_nan(a, b):
    return torch.all((a == b) | (torch.isnan(a) & torch.isnan(b)))


def get_dtype_for_format(data_format):
    """Get appropriate torch dtype for the given DataFormat"""
    return format_dict[data_format]


def create_test_tensors(data_format):
    """Create test tensors with appropriate dtype for the given format"""
    dtype = get_dtype_for_format(data_format)

    condition = torch.tensor([1, 0, -2, 0, 5, 0, 0, 8, 0, -1], dtype=dtype)
    condition_all_ones = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=dtype)
    condition_all_zeros = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dtype)

    # true and false value tensors
    true_values = torch.tensor(
        [
            1.0,
            float("nan"),
            3.0,
            float("inf"),
            -float("inf"),
            -1.0,
            0.0,
            -0.0,
            42.49,
            -92.42,
        ],
        dtype=dtype,
    )
    false_values = torch.tensor(
        [
            -1.0,
            999.9,
            float("nan"),
            -float("inf"),
            float("inf"),
            1.0,
            -0.0,
            0.0,
            -3.14,
            7.84,
        ],
        dtype=dtype,
    )

    return condition, condition_all_ones, condition_all_zeros, true_values, false_values


@parametrize(
    test_name="ttnn_where_test",
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float32,
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

    # Generate tensors dynamically based on current input format
    condition, condition_all_ones, condition_all_zeros, true_values, false_values = (
        create_test_tensors(formats.input_format)
    )
    dtype = get_dtype_for_format(formats.input_format)

    # Select test case
    if test_case == "mixed":
        test_condition = condition
    elif test_case == "all_ones":
        test_condition = condition_all_ones
    else:  # all_zeros
        test_condition = condition_all_zeros

    # Create test tensors with appropriate dtype for current format
    src_A = extend_tensor(test_condition.bool(), length=1024, dtype=dtype)
    src_B = extend_tensor(true_values, length=1024, dtype=dtype)
    src_C = extend_tensor(false_values, length=1024, dtype=dtype)

    core_loc = "0,0"
    buffer_A_address = 0x1A000
    buffer_B_address = 0x1B000
    buffer_C_address = 0x1C000

    if formats.input_format == DataFormat.Float32:
        pack_function_A = pack_fp32
        pack_function_B = pack_fp32
        pack_function_C = pack_fp32
    elif formats.input_format == DataFormat.Float16_b:
        pack_function_A = pack_bfp16
        pack_function_B = pack_bfp16
        pack_function_C = pack_bfp16
    else:
        raise ValueError(f"Unsupported input format: {formats.input_format}")

    golden = generate_golden(src_A, src_B, src_C)
    write_to_device(core_loc, buffer_A_address, pack_function_A(src_A))
    write_to_device(core_loc, buffer_B_address, pack_function_B(src_B))
    write_to_device(core_loc, buffer_C_address, pack_function_C(src_C))

    unpack_to_dest = formats.input_format.is_32_bit()

    test_config = {
        "formats": formats,
        "testname": test_name,
        "dest_acc": dest_acc,
        "mathop": mathop,
        "unpack_to_dest": unpack_to_dest,
    }

    run_test(test_config)

    wait_for_tensix_operations_finished()
    res_from_L1 = collect_results(formats, tile_count=1, address=0x1D000)
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

    dtype = get_dtype_for_format(formats.input_format)

    C = torch.arange(height * width, dtype=dtype)
    C = (C % 2).to(dtype)  # Alternates 0, 1, 0, 1, ... with correct dtype
    C = C.reshape(1, 1, height, width)
    C = C.expand(1, 1, height, width)  # Broadcast to (n, c, h, w)
    T = torch.ones(1, 1, height, width, dtype=dtype) * 2
    F = torch.ones(1, 1, height, width, dtype=dtype) * 11
    golden = torch.where(C != 0, T, F)

    C = C.flatten().to(format_dict[formats.input_format])
    T = T.flatten().to(format_dict[formats.input_format])
    F = F.flatten().to(format_dict[formats.input_format])

    core_loc = "0,0"
    buffer_A_address = 0x1A000
    buffer_B_address = 0x1B000
    buffer_C_address = 0x1C000

    if formats.input_format == DataFormat.Float32:
        pack_function_A = pack_fp32
        pack_function_B = pack_fp32
        pack_function_C = pack_fp32
    elif formats.input_format == DataFormat.Float16_b:
        pack_function_A = pack_bfp16
        pack_function_B = pack_bfp16
        pack_function_C = pack_bfp16
    else:
        raise ValueError(f"Unsupported input format: {formats.input_format}")

    golden = generate_golden(C, T, F)
    write_to_device(core_loc, buffer_A_address, pack_function_A(C))
    write_to_device(core_loc, buffer_B_address, pack_function_B(T))
    write_to_device(core_loc, buffer_C_address, pack_function_C(F))

    unpack_to_dest = formats.input_format.is_32_bit()

    test_config = {
        "formats": formats,
        "testname": test_name,
        "dest_acc": dest_acc,
        "unpack_to_dest": unpack_to_dest,
        "mathop": mathop,
    }

    run_test(test_config)

    wait_for_tensix_operations_finished()
    res_from_L1 = collect_results(formats, tile_count=1, address=0x1D000)
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
