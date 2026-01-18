# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark tests for comparing sequential vs parallel LayerNorm execution.
Run with: TT_METAL_DEVICE_PROFILER=1 python -m tracy -r -m pytest <this_file> -k <test_name>
"""

import pytest
import torch
import ttnn


@pytest.fixture
def setup_tensors():
    """Create common test tensors."""
    torch.manual_seed(42)
    batch_size, h, w = 1, 256, 128
    torch_input = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    torch_bias = torch.rand((w,), dtype=torch.bfloat16)
    return torch_input, torch_weight, torch_bias


def test_layernorm_single(device, setup_tensors):
    """
    Test 1: Single LayerNorm on interleaved tensor.
    This runs LayerNorm on a single tensor using default core allocation.
    """
    torch_input, torch_weight, torch_bias = setup_tensors

    # Move to device
    input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)
    weight_tensor = ttnn.from_torch(torch_weight, device=device, layout=ttnn.TILE_LAYOUT)
    bias_tensor = ttnn.from_torch(torch_bias, device=device, layout=ttnn.TILE_LAYOUT)

    # Run LayerNorm
    output = ttnn.layer_norm(input_tensor, epsilon=1e-5, weight=weight_tensor, bias=bias_tensor)
    ttnn.synchronize_device(device)

    # Verify output shape
    assert output.shape == input_tensor.shape


def test_layernorm_parallel(device, setup_tensors):
    """
    Test 2: Parallel LayerNorm with two branches on disjoint cores.
    This runs two LayerNorm operations in parallel - one on left cores, one on right cores.
    Total work is 2x the single test.
    """
    torch_input, torch_weight, torch_bias = setup_tensors

    # Define core ranges: left half (0-3) and right half (4-7) of the 8x8 grid
    left_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 7))])
    right_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 7))])

    # Create tensors for left branch
    input_left = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)
    weight_left = ttnn.from_torch(torch_weight, device=device, layout=ttnn.TILE_LAYOUT)
    bias_left = ttnn.from_torch(torch_bias, device=device, layout=ttnn.TILE_LAYOUT)

    # Create tensors for right branch (same data, but separate tensors)
    input_right = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)
    weight_right = ttnn.from_torch(torch_weight, device=device, layout=ttnn.TILE_LAYOUT)
    bias_right = ttnn.from_torch(torch_bias, device=device, layout=ttnn.TILE_LAYOUT)

    # Create branches
    branch_left = ttnn.parallel.branch(
        ttnn.layer_norm, input_left, cores=left_cores, epsilon=1e-5, weight=weight_left, bias=bias_left
    )
    branch_right = ttnn.parallel.branch(
        ttnn.layer_norm, input_right, cores=right_cores, epsilon=1e-5, weight=weight_right, bias=bias_right
    )

    # Execute in parallel
    results = ttnn.parallel([branch_left, branch_right])
    ttnn.synchronize_device(device)

    # Verify we got 2 outputs
    assert len(results) == 2
    assert results[0][0].shape == input_left.shape
    assert results[1][0].shape == input_right.shape
