# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_ax_plus_b_1D_tensors(device):
    """Test ax + b operation with 1D tensors"""
    torch.manual_seed(0)

    size = 8192

    # Create test tensors
    torch_tensor_a = torch.rand((size,), dtype=torch.bfloat16)  # coefficient 'a'
    torch_tensor_x = torch.rand((size,), dtype=torch.bfloat16)  # input 'x'
    torch_tensor_b = torch.rand((size,), dtype=torch.bfloat16)  # bias 'b'

    # Compute expected output: y = a * x + b
    torch_expected_output = torch_tensor_a * torch_tensor_x + torch_tensor_b

    # Convert to ttnn tensors
    ttnn_tensor_a = ttnn.from_torch(torch_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_tensor_x = ttnn.from_torch(torch_tensor_x, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_tensor_b = ttnn.from_torch(torch_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    # Execute ax + b operation
    ttnn_output = ttnn.prim.ax_plus_b(ttnn_tensor_a, ttnn_tensor_x, ttnn_tensor_b)
    ttnn_output = ttnn.to_torch(ttnn_output, torch_rank=1)

    # Verify the result
    assert_with_pcc(torch_expected_output, ttnn_output, 0.9999)
