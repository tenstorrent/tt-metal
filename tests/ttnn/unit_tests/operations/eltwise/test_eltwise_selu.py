# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import compare_equal_all_close
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp
from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops


def analyze_extreme_differences(input_tensor, calculated_tensor, expected_tensor, output_file="analysis_results.txt"):
    extreme_input_values = []

    with open(output_file, "w") as f:
        difference = expected_tensor - calculated_tensor

        min_val = torch.min(difference)
        max_val = torch.max(difference)
        min_indices = torch.nonzero(difference == min_val)
        max_indices = torch.nonzero(difference == max_val)

        torch_inp = ttnn.to_torch(input_tensor)

        def write_details_at_indices(indices, label):
            for idx in indices:
                idx_tuple = tuple(idx.tolist())

                input_val = torch_inp[idx_tuple].item()
                extreme_input_values.append(input_val)

                f.write(f"{label} at index {idx_tuple}:\n")
                f.write(f"  Input tensor value     : {input_tensor[idx_tuple]}\n")
                f.write(f"  Calculated tensor value: {calculated_tensor[idx_tuple].item()}\n")
                f.write(f"  Expected tensor value  : {expected_tensor[idx_tuple].item()}\n")
                f.write(
                    f"  Difference             : {(expected_tensor[idx_tuple] - calculated_tensor[idx_tuple]).item()}\n"
                )
                f.write("=" * 50 + "\n")

        f.write(f"\nMinimum difference: {min_val.item()}\n")
        write_details_at_indices(min_indices, "Minimum difference")

        f.write(f"\nMaximum difference: {max_val.item()}\n")
        write_details_at_indices(max_indices, "Maximum difference")

        f.write(f"\nInput values throwing max, min difference: {extreme_input_values}\n")
    print("\nInput values throwing max, min difference:", extreme_input_values)


def run_eltwise_selu_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)
    x = torch.Tensor(size=input_shape[0]).uniform_(-100, 100).to(torch.bfloat16)

    try:
        # get ref result
        ref_value = torch.nn.functional.selu(x)

        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])

        tt_result = ttnn.selu(x)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result, output_mem_config)

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape
    # analyze_extreme_differences(x, tt_result, ref_value, "my_diff_log.txt")
    comp_pass, _ = assert_with_pcc(ref_value, tt_result, 0.99)
    assert comp_pass and assert_with_ulp(ref_value, tt_result)


test_sweep_args = [
    (
        [(3, 2, 256, 256)],
        [ttnn.bfloat16],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        6861134,
    ),
    (
        [(12, 256, 256)],
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.DRAM_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        6411147,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_eltwise_selu(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_eltwise_selu_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "low, high",
    [
        (-100.0, 100.0),
        (
            -1.6 * 10**38,
            1.6 * 10**38,
        ),  # values higher than 1.7 to -3.3e+38, 3.3e+38 gives nan and hence not testing for that range
    ],
)
def test_unary_composite_selu_ttnn(input_shapes, low, high, device):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    torch_input = torch.linspace(high, low, num_elements, dtype=torch.bfloat16)
    torch_input = torch_input[:num_elements].reshape(input_shapes)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.selu(tt_in)
    result = ttnn.to_torch(output_tensor)
    golden_function = ttnn.get_golden_function(ttnn.selu)
    golden_tensor = golden_function(torch_input)
    # print_unique_values(golden_tensor - result)

    assert_with_pcc(golden_tensor, result, 0.99)


# values higher than 1.7 to -3.3e+38, 3.3e+38 gives nan, inf and hence not testing for that range
def test_unary_composite_selu_ttnn_ulp(device):
    num_elements = torch.prod(torch.tensor([3, 2, 1024, 1024])).item()
    torch_input = torch.linspace(1.6 * 10**38, -1.6 * 10**38, num_elements, dtype=torch.bfloat16)
    torch_input = torch_input[:num_elements].reshape([3, 2, 1024, 1024])

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.selu(tt_in)
    result = ttnn.to_torch(output_tensor)
    golden_function = ttnn.get_golden_function(ttnn.selu)
    golden_tensor = golden_function(torch_input)
    # print_unique_values(golden_tensor - result)

    comp_pass, _ = assert_with_pcc(golden_tensor, result, 0.99)
    assert comp_pass and assert_with_ulp(golden_tensor, result)


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 5, 5])),),
)
@pytest.mark.parametrize(
    "input_val, scale, alpha",
    [
        (0.3, 0.5, 1),
        (0.0, 1.0507, 1.6732),
        (1.0, 1.0507, 1.6732),
        (-1.0, 1.0507, 1.6732),
        (-5.0, 1.0507, 1.6732),
        (-0.5, 1.0, 1.5),
        (float("inf"), 1.0, float("inf")),
        (2.0, 1.2, 1.7),
        (float("-inf"), 1.0507, 1.6732),
        (float("inf"), 1.0507, 1.6732),
        (20.0, 1.0507, 1.6732),
        (-20.0, 1.0507, 1.6732),
        (-0.36719, 0.5, 1),
    ],
)
def test_selu_fill_val_bf16(input_shapes, input_val, scale, alpha, device):
    torch_input = torch.ones(input_shapes, dtype=torch.bfloat16) * input_val

    golden_function = ttnn.get_golden_function(ttnn.selu)
    golden = golden_function(torch_input, scale, alpha, device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.selu(tt_in, scale=scale, alpha=alpha)
    print("tt_in : ", tt_in)
    print("TT : ", tt_result)
    result = ttnn.to_torch(tt_result)
    print("GOLDEN : ", result)
    # print_unique_values(golden - result)

    comp_pass = compare_equal_all_close([golden], [result])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 1, 1])),),
)
@pytest.mark.parametrize(
    "input_val, scale, alpha",
    [
        (0.0, 1.0507, 1.6732),
        (1.0, 1.0507, 1.6732),
        (-5.0, 1.0507, 1.6732),
        (20.0, 1.0507, 1.6732),
        (-20.0, 1.0507, 1.6732),
    ],
)
def test_selu_fill_val_bf16_assert_with_ulp(input_shapes, input_val, scale, alpha, device):
    torch_input = torch.ones(input_shapes, dtype=torch.bfloat16) * input_val

    golden_function = ttnn.get_golden_function(ttnn.selu)
    golden = golden_function(torch_input, scale, alpha, device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.selu(tt_in, scale=scale, alpha=alpha)
    result = ttnn.to_torch(tt_result)

    assert assert_with_ulp(golden, result)


def debug_selu(x, alpha=1.67326, scale=1.0507):
    return torch.mul(
        scale,
        torch.add(
            torch.max(torch.full(x.shape, 0.0), x), torch.min(torch.full(x.shape, 0.0), alpha * (torch.exp(x) - 1.0))
        ),
    )


def test_debug_selu_bf16(device):
    torch_input = torch.tensor(
        [-0.20898, -0.18164, -0.69141, -2.42188, 63.75000, 98.50000, 86.50000, 97.00000, 100.00000, 91.00000],
        dtype=torch.bfloat16,
    )
    golden = debug_selu(torch_input)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.selu(tt_in)
    result = ttnn.to_torch(tt_result)
    print("TT RES : ", result)
    print("golden : ", golden)

    comp_pass, _ = assert_with_pcc(golden, result, 0.99)
    assert comp_pass and assert_with_ulp(golden, result)


def test_debug_selu_bf8(device):
    torch_input = torch.tensor([46.00000, 66.00000, 64.00000, 86.00000, 91.00000, -0.50000], dtype=torch.bfloat16)
    golden = debug_selu(torch_input)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat8_b,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.selu(tt_in)
    result = ttnn.to_torch(tt_result)
    print("TT RES : ", result)
    print("golden : ", golden)

    comp_pass, _ = assert_with_pcc(golden, result, 0.99)
    assert comp_pass and assert_with_ulp(golden, result)
