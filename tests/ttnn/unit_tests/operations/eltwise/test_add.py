# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("hw", [(32, 64)])
@pytest.mark.parametrize("scalar", [0.42])
def test_add_scalar(device, hw, scalar):
    torch_input_tensor_a = torch.rand(hw, dtype=torch.bfloat16)
    torch_output_tensor = scalar + torch_input_tensor_a

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    # Print tensor info during execution
    # if isinstance(input_tensor_a, ttnn.Tensor):
    #     print(f"\nTensor info for input_tensor_a:")
    #     print(f"  shape: {input_tensor_a.shape}")
    #     print(f"  dtype: {input_tensor_a.dtype}")
    #     print(f"  layout: {input_tensor_a.layout}")
    #     print(f"  memory_config: {input_tensor_a.memory_config()}")
    #     print(f"  storage_type: {input_tensor_a.storage_type()}")

    input_tensor_a = ttnn.to_memory_config(input_tensor_a, ttnn.L1_MEMORY_CONFIG)
    input_tensor_a = ttnn.to_memory_config(input_tensor_a, ttnn.DRAM_MEMORY_CONFIG)
    output = input_tensor_a + scalar

    # Print tensor info for result
    # if isinstance(output, ttnn.Tensor):
    #     print(f"\nTensor info for output:")
    #     print(f"  shape: {output.shape}")
    #     print(f"  dtype: {output.dtype}")
    #     print(f"  layout: {output.layout}")
    #     print(f"  memory_config: {output.memory_config()}")
    #     print(f"  storage_type: {output.storage_type()}")

    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)
