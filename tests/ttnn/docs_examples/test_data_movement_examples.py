# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch


def test_gather(device):
    # Create a tensor and an index tensor
    ttnn_input = ttnn.rand([4, 4], ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    ttnn_index = ttnn.rand([4, 2], ttnn.uint16, layout=ttnn.Layout.TILE, device=device)

    # Gather elements along dimension 1 using the index tensor
    ttnn.gather(ttnn_input, 1, index=ttnn_index)


def test_sort(device):
    # Create a tensor
    input_tensor = torch.Tensor([3, 1, 2])

    # Convert tensor to ttnn format
    input_tensor_ttnn = ttnn.from_torch(input_tensor, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)

    # Sort the tensor in ascending order
    sorted_tensor, indices = ttnn.sort(input_tensor_ttnn)

    # Sort the tensor in descending order
    sorted_tensor_desc, indices_desc = ttnn.sort(input_tensor_ttnn, descending=True)

    # Sort along a specific dimension
    input_tensor_2d = torch.Tensor([[3, 1, 2], [6, 5, 4]])
    input_tensor_2d_ttnn = ttnn.from_torch(input_tensor_2d, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    sorted_tensor_dim, indices_dim = ttnn.sort(input_tensor_2d_ttnn, dim=1)


def test_concat(device):
    # Create two tensors to concatenate
    tensor1 = ttnn.rand([1, 1, 64, 32], ttnn.bfloat16, device=device)
    tensor2 = ttnn.rand([1, 1, 64, 32], ttnn.bfloat16, device=device)

    # Concatenate along dimension 3
    concatenated_tensor = ttnn.concat([tensor1, tensor2], dim=3)
    print("Concatenated Tensor Shape:", concatenated_tensor.shape)  # [1, 1, 64, 64]
