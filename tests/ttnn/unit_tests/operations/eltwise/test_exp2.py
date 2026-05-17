# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import numpy as np
from tests.ttnn.utils_for_testing import assert_with_ulp, assert_allclose


def test_exp2_arange_masking(device):
    # Exp2 Working range - Overflow from 128(inf), Underflow till -127(<0)
    low = -126.0
    high = 127.0

    # Generate all possible bit patterns for bf16
    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    input_tensor = all_bitpatterns.view(torch.bfloat16)
    input_tensor_f32 = input_tensor.to(torch.float32)

    # masking to working range
    mask = (input_tensor_f32 >= low) & (input_tensor_f32 <= high)
    input_tensor = input_tensor[mask]

    tt_in = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.exp2)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.exp2(tt_in)
    result = ttnn.to_torch(tt_result)
    assert_with_ulp(golden, result, 1)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 2, 64, 120])),
        (torch.Size([1, 3, 320, 320])),
    ),
)
@pytest.mark.parametrize(
    "low, high",
    [
        (-5, 5),
        (-126, 127),
    ],
)
def test_exp2_ULP(input_shapes, low, high, device):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    torch_input = torch.linspace(high, low, num_elements, dtype=torch.bfloat16)
    torch_input = torch_input[:num_elements].reshape(input_shapes)

    golden_function = ttnn.get_golden_function(ttnn.exp2)
    golden = golden_function(torch_input, device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.exp2(tt_in)
    result = ttnn.to_torch(tt_result)
    assert_with_ulp(golden, result, 1)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 2, 64, 120])),
        (torch.Size([1, 3, 320, 320])),
    ),
)
@pytest.mark.parametrize(
    "low, high",
    [
        (-127, -126),
    ],
)
def test_exp2_atol(input_shapes, low, high, device):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    torch_input = torch.linspace(high, low, num_elements, dtype=torch.bfloat16)
    torch_input = torch_input[:num_elements].reshape(input_shapes)

    golden_function = ttnn.get_golden_function(ttnn.exp2)
    golden = golden_function(torch_input, device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.exp2(tt_in)
    result = ttnn.to_torch(tt_result)
    assert_allclose(tt_result, golden, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize(
    "tt_dtype, torch_dtype",
    [
        (ttnn.bfloat16, torch.bfloat16),
        (ttnn.float32, torch.float32),
    ],
)
def test_exp2_special_values(tt_dtype, torch_dtype, device):
    special_inputs = [
        0.0,
        1.0,
        -1.0,
        10.0,
        -10.0,
        127.0,
        -126.0,
        float("inf"),
        float("-inf"),
        float("nan"),
        128.0,
        -127.0,
    ]

    pad_count = 32 * 32 - len(special_inputs)
    torch_input = torch.tensor(special_inputs + [0.0] * pad_count, dtype=torch_dtype).reshape(1, 1, 32, 32)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=tt_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    result = ttnn.to_torch(ttnn.exp2(tt_in)).reshape(-1)

    expected = torch.exp2(torch_input.to(torch.float32)).reshape(-1)
    expected[8] = 0.0
    expected[10] = float("inf")
    expected[11] = 0.0

    finite_indices = []
    for index, value in enumerate(special_inputs):
        actual_value = result[index].item()
        expected_value = expected[index].item()

        if np.isnan(expected_value):
            assert np.isnan(actual_value), f"exp2({value}): expected NaN, got {actual_value}"
        elif np.isinf(expected_value):
            assert np.isinf(actual_value) and np.sign(actual_value) == np.sign(expected_value), (
                f"exp2({value}): expected {expected_value}, got {actual_value}"
            )
        else:
            finite_indices.append(index)

    if finite_indices:
        indices = torch.tensor(finite_indices)
        assert_with_ulp(expected[indices], result[indices].to(torch.float32), 1)
