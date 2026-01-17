# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.use_module_device


def torch_rms_norm(x, gamma=None, eps=1e-5):
    """Reference RMS norm implementation in PyTorch."""
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    x_normed = x / rms
    if gamma is not None:
        x_normed = x_normed * gamma
    return x_normed


def torch_layer_norm(x, gamma=None, beta=None, residual=None, eps=1e-5):
    """Reference Layer norm implementation in PyTorch with optional residual."""
    # Add residual if provided
    if residual is not None:
        x = x + residual

    # Compute mean and variance over the last dimension
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)

    # Normalize
    x_normed = (x - mean) / torch.sqrt(var + eps)

    # Apply gamma (scale) and beta (shift)
    if gamma is not None:
        x_normed = x_normed * gamma
    if beta is not None:
        x_normed = x_normed + beta

    return x_normed


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [128])  # Larger height for more tile rows
@pytest.mark.parametrize("w", [64])
def test_parallel_rms_norm(device, batch_size, h, w):
    """Test parallel execution using ttnn.parallel.branch() API."""
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
    # Use single core per branch on different rows (y-axis) which works correctly
    cores_a = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
    cores_b = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 0))])

    # Create branches using ttnn.parallel.branch(operation, *args, cores=..., **kwargs)
    branch_a = ttnn.parallel.branch(
        ttnn.rms_norm,
        input_a,
        cores=cores_a,
        epsilon=1e-5,
        weight=weight_a,
    )
    branch_b = ttnn.parallel.branch(
        ttnn.rms_norm,
        input_b,
        cores=cores_b,
        epsilon=1e-5,
        weight=weight_b,
    )

    # Execute in parallel using ttnn.parallel([...])
    results = ttnn.parallel([branch_a, branch_b])

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
    """Test parallel RMS norm with different random seed."""
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

    # Use single core per branch on different rows (y-axis) which works correctly
    cores_a = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))])
    cores_b = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(0, 3))])

    # Using ttnn.parallel.branch() API
    branch_a = ttnn.parallel.branch(ttnn.rms_norm, input_a, cores=cores_a, epsilon=1e-5, weight=weight_a)
    branch_b = ttnn.parallel.branch(ttnn.rms_norm, input_b, cores=cores_b, epsilon=1e-5, weight=weight_b)

    results = ttnn.parallel([branch_a, branch_b])

    output_a = ttnn.to_torch(ttnn.from_device(results[0][0]))
    output_b = ttnn.to_torch(ttnn.from_device(results[1][0]))

    assert_with_pcc(torch_output_a, output_a, 0.999)
    assert_with_pcc(torch_output_b, output_b, 0.999)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [128])  # 4 tile rows (128/32)
@pytest.mark.parametrize("w", [64])  # 2 tile columns (64/32)
def test_rms_full_grid(device, batch_size, h, w):
    """
    Test parallel execution using full 8x8 compute grid split into 4x4 blocks of 2x2 cores.
    Each 2x2 block runs a sharded RMS norm calculation.
    Total: 16 parallel branches.
    """
    torch.manual_seed(123)

    # Grid layout: 8x8 cores split into 4x4 blocks of 2x2 cores
    # Block (i,j) covers cores [(2*i, 2*j), (2*i+1, 2*j+1)]
    num_blocks_x = 4  # 8 cores / 2 cores per block
    num_blocks_y = 4
    num_branches = num_blocks_x * num_blocks_y  # 16 branches

    # Create input and weight tensors for each branch
    torch_inputs = []
    torch_weights = []
    torch_outputs = []

    for i in range(num_branches):
        torch_input = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
        torch_weight = torch.rand((w,), dtype=torch.bfloat16)
        torch_output = torch_rms_norm(torch_input, torch_weight)

        torch_inputs.append(torch_input)
        torch_weights.append(torch_weight)
        torch_outputs.append(torch_output)

    # Move tensors to device
    device_inputs = [ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT) for t in torch_inputs]
    device_weights = [ttnn.from_torch(t, device=device, layout=ttnn.TILE_LAYOUT) for t in torch_weights]

    # Create branches for each 2x2 block
    branches = []
    for block_y in range(num_blocks_y):
        for block_x in range(num_blocks_x):
            branch_idx = block_y * num_blocks_x + block_x

            # Calculate core range for this 2x2 block
            start_x = block_x * 2
            start_y = block_y * 2
            end_x = start_x + 1  # 2 cores wide (0,1)
            end_y = start_y + 1  # 2 cores tall (0,1)

            core_range = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(start_x, start_y), ttnn.CoreCoord(end_x, end_y))]
            )

            # Create branch for this block
            branch = ttnn.parallel.branch(
                ttnn.rms_norm,
                device_inputs[branch_idx],
                cores=core_range,
                epsilon=1e-5,
                weight=device_weights[branch_idx],
            )
            branches.append(branch)

    # Execute all 16 branches in parallel
    results = ttnn.parallel(branches)

    # Verify each branch output against torch reference
    for i in range(num_branches):
        output = ttnn.to_torch(ttnn.from_device(results[i][0]))
        assert_with_pcc(torch_outputs[i], output, 0.999)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("num_branches", [2])
def test_layernorm_parallel_interleaved(device, batch_size, h, w, num_branches):
    """
    Test parallel LayerNorm with interleaved (non-sharded) tensors.
    Uses the simpler multi-core program factory to test parallel infrastructure.
    """
    torch.manual_seed(456)

    # Create input, weight, bias, and residual tensors for each branch
    torch_inputs = []
    torch_weights = []
    torch_biases = []
    torch_residuals = []
    torch_outputs = []

    for i in range(num_branches):
        torch_input = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
        torch_weight = torch.rand((w,), dtype=torch.bfloat16)
        torch_bias = torch.rand((w,), dtype=torch.bfloat16)
        torch_residual = torch.rand((batch_size, h, w), dtype=torch.bfloat16)

        # Compute reference: LayerNorm with residual
        torch_output = torch_layer_norm(torch_input, torch_weight, torch_bias, torch_residual)

        torch_inputs.append(torch_input)
        torch_weights.append(torch_weight)
        torch_biases.append(torch_bias)
        torch_residuals.append(torch_residual)
        torch_outputs.append(torch_output)

    # Create branches with different core ranges (non-overlapping single cores)
    branches = []

    for i in range(num_branches):
        # Each branch gets its own core on different rows (y-axis) which works correctly
        core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, i), ttnn.CoreCoord(0, i))})

        # Move tensors to device (interleaved memory)
        input_tensor = ttnn.from_torch(
            torch_inputs[i],
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        residual_tensor = ttnn.from_torch(
            torch_residuals[i],
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        weight_tensor = ttnn.from_torch(
            torch_weights[i],
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )
        bias_tensor = ttnn.from_torch(
            torch_biases[i],
            device=device,
            layout=ttnn.TILE_LAYOUT,
        )

        # Create branch for this core using layer_norm
        branch = ttnn.parallel.branch(
            ttnn.layer_norm,
            input_tensor,
            cores=core_range_set,
            epsilon=1e-5,
            weight=weight_tensor,
            bias=bias_tensor,
            residual_input_tensor=residual_tensor,
        )
        branches.append(branch)

    # Execute all branches in parallel
    results = ttnn.parallel(branches)

    # Verify each branch output against torch reference
    for i in range(num_branches):
        output = ttnn.to_torch(ttnn.from_device(results[i][0]))
        assert_with_pcc(torch_outputs[i], output, 0.999)
