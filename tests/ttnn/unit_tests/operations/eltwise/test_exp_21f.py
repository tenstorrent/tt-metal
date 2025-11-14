# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import numpy as np
from tests.ttnn.utils_for_testing import assert_with_ulp, assert_allclose


def test_exp_arange_masking(device):
    # Exp Working range - Overflow from 88.5(inf), Underflow till -87(<0)
    low = -87.0
    high = 88.5

    # Generate all possible bit pattersn for bf16
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

    golden_function = ttnn.get_golden_function(ttnn.exp)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.exp(tt_in)
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
        (-87.0, 88.5),
    ],
)
def test_exp_ULP(input_shapes, low, high, device):
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
    "low, high, testing_dtype, expected_rtol, expected_atol",
    [
        (-89, -87, "bfloat16", 1e-2, 1e-3),
        (-87.3, 88.7, "float32", 1e-2, 1e-3),
    ],
)
def test_exp_atol(input_shapes, low, high, testing_dtype, expected_rtol, expected_atol, device):
    torch_dtype = getattr(torch, testing_dtype)
    ttnn_dtype = getattr(ttnn, testing_dtype)

    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    torch_input = torch.linspace(high, low, num_elements, dtype=torch_dtype)
    torch_input = torch_input[:num_elements].reshape(input_shapes)

    golden_function = ttnn.get_golden_function(ttnn.exp)
    golden = golden_function(torch_input, device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.exp(tt_in)
    assert_allclose(tt_result, golden, rtol=expected_rtol, atol=expected_atol)


def test_exp_fp32(device):
    # Exp Working range for fp32 - Overflow from 88.7(inf), Underflow till -87.3(<0)
    low = -87.3
    high = 88.7

    all_bitpatterns = torch.arange(0, 2**16, dtype=torch.int32).to(torch.uint16)
    input_tensor = all_bitpatterns.view(torch.bfloat16)
    input_tensor = input_tensor.to(torch.float32)

    # masking to working range
    mask = (input_tensor >= low) & (input_tensor <= high)
    input_tensor = input_tensor[mask]

    # Mask NaN
    mask = torch.isnan(input_tensor)
    input_tensor[mask] = 1.0

    tt_in = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.exp)
    golden = golden_function(input_tensor, device=device)

    tt_result = ttnn.exp(tt_in)
    result = ttnn.to_torch(tt_result)

    assert_with_ulp(golden, result, 134, allow_nonfinite=True)
