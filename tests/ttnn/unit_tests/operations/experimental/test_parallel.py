# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

# Access the experimental parallel operations from the correct module path
experimental_ops = ttnn._ttnn.operations.experimental

pytestmark = pytest.mark.use_module_device


def torch_rms_norm(x, gamma=None, eps=1e-5):
    """Reference RMS norm implementation in PyTorch."""
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    x_normed = x / rms
    if gamma is not None:
        x_normed = x_normed * gamma
    return x_normed


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [128])  # Larger height for more tile rows
@pytest.mark.parametrize("w", [64])
def test_parallel_two_rms_norm(device, batch_size, h, w):
    """Test parallel execution of two RMS norm operations on disjoint cores."""
    torch.manual_seed(0)

    # Create two input tensors
    torch_input_a = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_input_b = torch.rand((batch_size, h, w), dtype=torch.bfloat16)

    # Create weights (gamma)
    torch_weight_a = torch.rand((w,), dtype=torch.bfloat16)
    torch_weight_b = torch.rand((w,), dtype=torch.bfloat16)

    # Compute reference outputs
    torch_output_a = torch_rms_norm(torch_input_a, torch_weight_a)
    torch_output_b = torch_rms_norm(torch_input_b, torch_weight_b)

    # Move tensors to device
    input_a = ttnn.from_torch(torch_input_a, device=device, layout=ttnn.TILE_LAYOUT)
    input_b = ttnn.from_torch(torch_input_b, device=device, layout=ttnn.TILE_LAYOUT)
    weight_a = ttnn.from_torch(torch_weight_a, device=device, layout=ttnn.TILE_LAYOUT)
    weight_b = ttnn.from_torch(torch_weight_b, device=device, layout=ttnn.TILE_LAYOUT)

    # Define disjoint core ranges for each branch
    # Use smaller core ranges (2 cores each) to ensure enough work per core
    # 128 height = 4 tile rows, 2 cores = 2 tile rows per core
    cores_a = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    # Second set of cores for branch B
    cores_b = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 0))])

    # Create branches
    branch_a = experimental_ops.rms_norm_branch(
        cores=cores_a,
        input=input_a,
        epsilon=1e-5,
        weight=weight_a,
    )
    branch_b = experimental_ops.rms_norm_branch(
        cores=cores_b,
        input=input_b,
        epsilon=1e-5,
        weight=weight_b,
    )

    # Execute in parallel
    results = experimental_ops.parallel([branch_a, branch_b])

    # Extract outputs
    output_a = ttnn.from_device(results[0][0])
    output_b = ttnn.from_device(results[1][0])

    # Convert back to torch
    output_a = ttnn.to_torch(output_a)
    output_b = ttnn.to_torch(output_b)

    # Verify results
    assert_with_pcc(torch_output_a, output_a, 0.999)
    assert_with_pcc(torch_output_b, output_b, 0.999)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [128])  # Same as first test
@pytest.mark.parametrize("w", [64])
def test_parallel_rms_norm_different_seed(device, batch_size, h, w):
    """Test parallel RMS norm with larger dimensions."""
    torch.manual_seed(42)

    torch_input_a = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_input_b = torch.rand((batch_size, h, w), dtype=torch.bfloat16)

    # Create weights (gamma) - RMS norm works better with weights
    torch_weight_a = torch.rand((w,), dtype=torch.bfloat16)
    torch_weight_b = torch.rand((w,), dtype=torch.bfloat16)

    torch_output_a = torch_rms_norm(torch_input_a, torch_weight_a)
    torch_output_b = torch_rms_norm(torch_input_b, torch_weight_b)

    input_a = ttnn.from_torch(torch_input_a, device=device, layout=ttnn.TILE_LAYOUT)
    input_b = ttnn.from_torch(torch_input_b, device=device, layout=ttnn.TILE_LAYOUT)
    weight_a = ttnn.from_torch(torch_weight_a, device=device, layout=ttnn.TILE_LAYOUT)
    weight_b = ttnn.from_torch(torch_weight_b, device=device, layout=ttnn.TILE_LAYOUT)

    # Use smaller core ranges (2 cores each) to ensure enough work per core
    cores_a = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    cores_b = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 0))])

    branch_a = experimental_ops.rms_norm_branch(
        cores=cores_a,
        input=input_a,
        epsilon=1e-5,
        weight=weight_a,
    )
    branch_b = experimental_ops.rms_norm_branch(
        cores=cores_b,
        input=input_b,
        epsilon=1e-5,
        weight=weight_b,
    )

    results = experimental_ops.parallel([branch_a, branch_b])

    output_a = ttnn.to_torch(ttnn.from_device(results[0][0]))
    output_b = ttnn.to_torch(ttnn.from_device(results[1][0]))

    assert_with_pcc(torch_output_a, output_a, 0.999)
    assert_with_pcc(torch_output_b, output_b, 0.999)
