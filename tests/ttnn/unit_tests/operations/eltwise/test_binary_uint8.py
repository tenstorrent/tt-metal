# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import torch
import pytest
import ttnn
from models.common.utility_functions import is_blackhole

pytestmark = pytest.mark.use_module_device


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
    "low_a, high_a, low_b, high_b",
    [
        (0, 255, 0, 255),
    ],
)
@pytest.mark.parametrize(
    "ttnn_op",
    [ttnn.ne],
)
def test_binary_relational_uint8(a_shape, b_shape, low_a, high_a, low_b, high_b, ttnn_op, device):
    if is_blackhole() and os.environ.get("TT_METAL_SIMULATOR"):
        pytest.skip("Skipping on tt-sim in Blackhole/Wormhole B0")
    num_elements = max(int(torch.prod(torch.tensor(a_shape)).item()), 1)
    torch_input_tensor_a = torch.linspace(high_a, low_a, num_elements, dtype=torch.int32)
    corner_cases = torch.tensor([0, 1, 255], dtype=torch.int32)
    torch_input_tensor_a = torch.cat([torch_input_tensor_a, corner_cases])
    torch_input_tensor_a = torch_input_tensor_a[-num_elements:].reshape(a_shape)

    num_elements = max(int(torch.prod(torch.tensor(b_shape)).item()), 1)
    torch_input_tensor_b = torch.linspace(high_b, low_b, num_elements, dtype=torch.int32)
    corner_cases = torch.tensor([0, 1, 255], dtype=torch.int32)
    torch_input_tensor_b = torch.cat([torch_input_tensor_b, corner_cases])
    torch_input_tensor_b = torch_input_tensor_b[-num_elements:].reshape(b_shape)

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

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
