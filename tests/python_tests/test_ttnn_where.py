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
from helpers.pack import pack_bfp8_b, pack_bfp16, pack_fp32
from helpers.param_config import (
    clean_params,
    generate_param_ids,
    generate_params,
    input_output_formats,
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


# SUPPORTED FORMATS FOR TEST - allow sweeping through multiple formats
supported_formats = [
    DataFormat.Float16_b,
    DataFormat.Float32,
]


def get_dtype_for_format(data_format):
    """Get appropriate torch dtype for the given DataFormat"""
    return format_dict[data_format]


def get_dest_acc_for_format(data_format):
    """Get appropriate dest_acc options for the given DataFormat"""
    if data_format == DataFormat.Float32:
        return [DestAccumulation.Yes]  # Float32 requires dest_acc=Yes
    elif data_format == DataFormat.Float16_b:
        return [DestAccumulation.No]  # Float16_b only allows dest_acc=No
    else:
        return [DestAccumulation.No, DestAccumulation.Yes]  # Other formats can use both


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


# Generate parameter combinations that dynamically include appropriate dest_acc for each format
# Use same=True to ensure input and output formats are identical (no mixing)
test_formats = input_output_formats(supported_formats, same=True)
all_params = []
for fmt in test_formats:
    dest_acc_options = get_dest_acc_for_format(fmt.input_format)
    # Use generate_params to create properly formatted parameter tuples
    params = generate_params(
        ["ttnn_where_test"],
        [fmt],
        dest_acc=dest_acc_options,
        mathop=[MathOperation.TTNNWhere],
    )
    all_params.extend(params)

param_ids = generate_param_ids(all_params)


@pytest.mark.parametrize(
    "testname, formats, dest_acc, mathop",
    clean_params(all_params),
    ids=param_ids,
)
@pytest.mark.parametrize("test_case", ["mixed", "all_ones", "all_zeros"])
def test_ttnn_where(testname, formats, dest_acc, mathop, test_case):

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

    # Skipping test combinations that are not supported
    if (
        formats.output_format == DataFormat.Float32
        or formats.input_format == DataFormat.Float32
    ) and dest_acc == DestAccumulation.No:
        pytest.skip(
            "Skipping test for Float32 input format with NO dest_acc, as it is not supported."
        )

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
    elif formats.input_format == DataFormat.Bfp8_b:
        pack_function_A = pack_bfp8_b
        pack_function_B = pack_bfp8_b
        pack_function_C = pack_bfp8_b
    else:
        raise ValueError(f"Unsupported input format: {formats.input_format}")

    golden = generate_golden(src_A, src_B, src_C)
    write_to_device(core_loc, buffer_A_address, pack_function_A(src_A))
    write_to_device(core_loc, buffer_B_address, pack_function_B(src_B))
    write_to_device(core_loc, buffer_C_address, pack_function_C(src_C))

    unpack_to_dest = formats.input_format.is_32_bit()

    test_config = {
        "formats": formats,
        "testname": testname,
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
            if formats.output_format
            in [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]
            else torch.bfloat16
        ),
    )
    res_tensor = torch.tensor(
        res_from_L1,
        dtype=(
            format_dict[formats.output_format]
            if formats.output_format
            in [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]
            else torch.bfloat16
        ),
    )

    assert torch_equal_nan(golden_tensor, res_tensor)


# MCW test with dynamic format sweeping like main test
# Use same input/output format - no mixing
test_formats_mcw = input_output_formats(supported_formats, same=True)
all_params_mcw = []
for fmt in test_formats_mcw:
    dest_acc_options = get_dest_acc_for_format(fmt.input_format)
    # Use generate_params to create properly formatted parameter tuples
    params = generate_params(
        ["ttnn_where_test"],
        [fmt],
        dest_acc=dest_acc_options,
        mathop=[MathOperation.TTNNWhere],
    )
    all_params_mcw.extend(params)

param_ids_mcw = generate_param_ids(all_params_mcw)


@pytest.mark.parametrize(
    "testname, formats, dest_acc, mathop",
    clean_params(all_params_mcw),
    ids=param_ids_mcw,
)
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [32])
def test_ttnn_where_mcw(testname, formats, dest_acc, mathop, h, w):
    # Generate dtype dynamically based on current input format
    dtype = get_dtype_for_format(formats.input_format)

    C = torch.arange(h * w, dtype=dtype)
    C = (C % 2).to(dtype)  # Alternates 0, 1, 0, 1, ... with correct dtype
    C = C.reshape(1, 1, h, w)
    C = C.expand(1, 1, h, w)  # Broadcast to (n, c, h, w)
    T = torch.ones(1, 1, h, w, dtype=dtype) * 2
    F = torch.ones(1, 1, h, w, dtype=dtype) * 11
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
    elif formats.input_format == DataFormat.Bfp8_b:
        pack_function_A = pack_bfp8_b
        pack_function_B = pack_bfp8_b
        pack_function_C = pack_bfp8_b
    else:
        raise ValueError(f"Unsupported input format: {formats.input_format}")

    golden = generate_golden(C, T, F)
    write_to_device(core_loc, buffer_A_address, pack_function_A(C))
    write_to_device(core_loc, buffer_B_address, pack_function_B(T))
    write_to_device(core_loc, buffer_C_address, pack_function_C(F))

    unpack_to_dest = formats.input_format.is_32_bit()

    test_config = {
        "formats": formats,
        "testname": testname,
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
            if formats.output_format
            in [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]
            else torch.bfloat16
        ),
    )

    golden_tensor = golden_tensor.flatten()[:1024]  # Ensure it matches the result size

    res_tensor = torch.tensor(
        res_from_L1,
        dtype=(
            format_dict[formats.output_format]
            if formats.output_format
            in [DataFormat.Float16, DataFormat.Float16_b, DataFormat.Float32]
            else torch.bfloat16
        ),
    )

    assert len(res_tensor) == len(golden_tensor)
    assert torch_equal_nan(golden_tensor, res_tensor)
