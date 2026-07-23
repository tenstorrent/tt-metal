# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import torch
import pytest
import ttnn
from models.common.utility_functions import is_blackhole


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        (torch.Size([1, 2, 32]), torch.Size([1, 2, 32])),
        (torch.Size([1]), torch.Size([1, 5, 12])),
        (torch.Size([1, 2, 32, 64, 125]), torch.Size([1, 2, 32, 1, 1])),
        (torch.Size([]), torch.Size([])),
        (torch.Size([5]), torch.Size([1])),
    ],
)
@pytest.mark.parametrize(
    "ttnn_op",
    [ttnn.ne, ttnn.eq, ttnn.lt, ttnn.gt, ttnn.le, ttnn.ge],
)
def test_binary_relational_uint8(a_shape, b_shape, ttnn_op, device):
    low, high = 0, 255
    num_elements = max(int(torch.prod(torch.tensor(a_shape)).item()), 1)
    torch_input_tensor_a = torch.linspace(high, low, num_elements, dtype=torch.int32)
    corner_cases = torch.tensor([0, 1, 255], dtype=torch.int32)
    torch_input_tensor_a = torch.cat([torch_input_tensor_a, corner_cases])
    torch_input_tensor_a = torch_input_tensor_a[-num_elements:].reshape(a_shape)

    num_elements = max(int(torch.prod(torch.tensor(b_shape)).item()), 1)
    torch_input_tensor_b = torch.linspace(high, low, num_elements, dtype=torch.int32)
    corner_cases = torch.tensor([0, 1, 255], dtype=torch.int32)
    torch_input_tensor_b = torch.cat([torch_input_tensor_b, corner_cases])
    torch_input_tensor_b = torch_input_tensor_b[-num_elements:].reshape(b_shape)

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device).to(torch.uint8)
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint8,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint8,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn_op(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.uint8)

    assert torch.equal(output_tensor, torch_output_tensor)


@pytest.mark.parametrize("shape", [torch.Size([1, 2, 32]), torch.Size([2, 64, 64])])
@pytest.mark.parametrize("scalar", [0, 1, 128, 255])
@pytest.mark.parametrize(
    "ttnn_op",
    [ttnn.lt, ttnn.gt, ttnn.le, ttnn.ge, ttnn.ne, ttnn.eq],
)
def test_binary_relational_uint8_tensor_scalar(shape, scalar, ttnn_op, device):
    # scalar must be in [0, 255] to stay within the UINT8 value range
    num_elements = int(torch.prod(torch.tensor(shape)).item())
    torch_input = torch.arange(num_elements, dtype=torch.int32).remainder(256).reshape(shape)

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output = golden_function(torch_input, scalar, device=device).to(torch.uint8)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.uint8,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn_op(input_tensor, float(scalar))
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.uint8)

    assert torch.equal(output_tensor, torch_output)
