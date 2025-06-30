# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import compare_equal
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp


def print_unique_values(tensor):
    unique_values = torch.unique(tensor.to(torch.float32))
    unique_array = unique_values.numpy()
    print("Unique values:", unique_array)
    print("Min value:", torch.min(tensor).item())
    print("Max value:", torch.max(tensor).item())


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize("input_val", [-88, -89, -90])
def test_unary_max_fill_val_bf16(input_shapes, input_val, device):
    torch_input = torch.ones(input_shapes, dtype=torch.bfloat16) * input_val

    golden_function = ttnn.get_golden_function(ttnn.exp)
    golden = golden_function(torch_input, device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.exp(tt_in)
    result = ttnn.to_torch(tt_result)
    print("\nInput : ", torch_input[0, 0, 0, 0])
    print("\ngolden:", golden[0, 0, 0, 0], "\nTTNN:", result[0, 0, 0, 0])
    # assert_with_pcc(golden, result, 0.999)
    assert compare_equal([tt_result], [golden])


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        # (torch.Size([1, 2, 64, 120])),
        # (torch.Size([1, 3, 320, 320])),
    ),
)
@pytest.mark.parametrize(
    "low, high",
    [
        (-100, 88),
    ],
)
def test_unary_max_bf16(input_shapes, low, high, device):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    torch_input = torch.linspace(high, low, num_elements, dtype=torch.bfloat16)
    torch_input = torch_input[:num_elements].reshape(input_shapes)

    golden_function = ttnn.get_golden_function(ttnn.exp)
    golden = golden_function(torch_input, device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.exp(tt_in)
    result = ttnn.to_torch(tt_result)
    # assert_with_pcc(golden, result, 0.999)
    print("\ngolden:", golden, "\nTTNN:", result)
    print_unique_values(golden - result)
    assert_with_ulp(golden, result)


def test_pow_bf16(device):
    torch_input_a = torch.tensor([9.0, 100000])
    torch_input_b = torch.tensor([2.0, 1.7984])

    golden_function = ttnn.get_golden_function(ttnn.pow)
    golden = golden_function(torch_input_a, torch_input_b, device=device)

    tt_ina = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_inb = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.pow(tt_ina, tt_inb)
    result = ttnn.to_torch(tt_result)
    # assert_with_pcc(golden, result, 0.999)
    print("\ngolden:", golden, "\nTTNN:", result)
    assert_with_ulp(golden, result)
