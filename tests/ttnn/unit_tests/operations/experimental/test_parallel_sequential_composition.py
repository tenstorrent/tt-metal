# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for composing ttnn.parallel and ttnn.sequential operations.

This tests nested execution patterns like:
- Parallel containing sequential
- Sequential containing parallel
- Deeply nested combinations

Example structure tested:
    ttnn.parallel([
        branch_A: simple LayerNorm,
        branch_B: ttnn.sequential([
            step_1: RMS norm,
            step_2: ttnn.parallel([
                sub_branch_1: LayerNorm,
                sub_branch_2: LayerNorm
            ])
        ])
    ])
"""

import pytest
import torch
import ttnn


def torch_rms_norm(x, gamma=None, eps=1e-5):
    """PyTorch reference implementation of RMS normalization."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    if gamma is not None:
        x_normed = x_normed * gamma
    return x_normed


def torch_layer_norm(x, gamma=None, beta=None, eps=1e-5):
    """PyTorch reference implementation of LayerNorm."""
    mean = x.mean(-1, keepdim=True)
    variance = x.var(-1, keepdim=True, unbiased=False)
    x_normed = (x - mean) / torch.sqrt(variance + eps)
    if gamma is not None:
        x_normed = x_normed * gamma
    if beta is not None:
        x_normed = x_normed + beta
    return x_normed


def assert_with_pcc(expected, actual, pcc=0.99):
    """Assert that actual matches expected within PCC tolerance."""
    expected_flat = expected.flatten().float()
    actual_flat = actual.flatten().float()
    correlation = torch.corrcoef(torch.stack([expected_flat, actual_flat]))[0, 1]
    assert correlation >= pcc, f"PCC {correlation} < {pcc}"


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sequential_containing_parallel(device, batch_size, h, w):
    """
    Test: Sequential operation that contains a parallel operation.

    Structure:
        ttnn.sequential([
            (ttnn.rms_norm, input1),           # Step 1: RMS norm
            (ttnn.parallel, [branch_a, branch_b])  # Step 2: Parallel of 2 LayerNorms
        ])

    Note: Since ttnn.sequential just calls operations in order,
    we pass ttnn.parallel as an operation with its branches as the argument.
    """
    torch.manual_seed(42)

    # Create test tensors
    torch_input1 = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_input2a = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_input2b = torch.rand((batch_size, h, w), dtype=torch.bfloat16)

    # Compute torch reference
    torch_output1 = torch_rms_norm(torch_input1, eps=1e-5)
    torch_output2a = torch_layer_norm(torch_input2a, eps=1e-6)
    torch_output2b = torch_layer_norm(torch_input2b, eps=1e-6)

    # Define core ranges for parallel step
    left_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])
    right_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 3))])

    # Move tensors to device
    input1 = ttnn.from_torch(torch_input1, device=device, layout=ttnn.TILE_LAYOUT)
    input2a = ttnn.from_torch(torch_input2a, device=device, layout=ttnn.TILE_LAYOUT)
    input2b = ttnn.from_torch(torch_input2b, device=device, layout=ttnn.TILE_LAYOUT)

    # Create parallel branches for step 2
    branch_a = ttnn.parallel.branch(ttnn.layer_norm, input2a, cores=left_cores, epsilon=1e-6)
    branch_b = ttnn.parallel.branch(ttnn.layer_norm, input2b, cores=right_cores, epsilon=1e-6)

    # Execute sequential: Step 1 is RMS norm, Step 2 is parallel LayerNorm
    results = ttnn.sequential(
        [
            (ttnn.rms_norm, input1, {"epsilon": 1e-5}),
            (ttnn.parallel, [branch_a, branch_b]),
        ]
    )

    # Verify Step 1: RMS norm result
    output1 = ttnn.to_torch(ttnn.from_device(results[0]))
    assert_with_pcc(torch_output1, output1, 0.999)

    # Verify Step 2: Parallel LayerNorm results
    parallel_results = results[1]
    output2a = ttnn.to_torch(ttnn.from_device(parallel_results[0][0]))
    output2b = ttnn.to_torch(ttnn.from_device(parallel_results[1][0]))
    assert_with_pcc(torch_output2a, output2a, 0.999)
    assert_with_pcc(torch_output2b, output2b, 0.999)

    print("✓ test_sequential_containing_parallel passed!")


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_parallel_with_sequential_branch(device, batch_size, h, w):
    """
    Test: Parallel operation where one branch runs a sequential of operations.

    Structure:
        ttnn.parallel([
            branch_A: LayerNorm (simple),
            branch_B: result of sequential [RMS norm, then another op]
        ])

    Note: The sequential runs first to produce branch_B's input processing,
    then we use the final result as the "branch" input.
    Actually, since parallel branches are pre-defined, we run sequential first
    and use its output tensor for branch_B.
    """
    torch.manual_seed(42)

    # Create test tensors
    torch_input_a = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_input_b = torch.rand((batch_size, h, w), dtype=torch.bfloat16)

    # Compute torch reference for branch A (simple LayerNorm)
    torch_output_a = torch_layer_norm(torch_input_a, eps=1e-5)

    # Compute torch reference for branch B (sequential: RMS then LayerNorm)
    torch_intermediate_b = torch_rms_norm(torch_input_b, eps=1e-6)
    torch_output_b = torch_layer_norm(torch_intermediate_b, eps=1e-6)

    # Define core ranges
    left_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])
    right_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 3))])

    # Move tensors to device
    input_a = ttnn.from_torch(torch_input_a, device=device, layout=ttnn.TILE_LAYOUT)
    input_b = ttnn.from_torch(torch_input_b, device=device, layout=ttnn.TILE_LAYOUT)

    # Branch A: Simple LayerNorm
    branch_a = ttnn.parallel.branch(ttnn.layer_norm, input_a, cores=left_cores, epsilon=1e-5)

    # Branch B: First run sequential to process input, then create branch from result
    # Sequential: RMS norm -> LayerNorm
    sequential_results = ttnn.sequential(
        [
            (ttnn.rms_norm, input_b, {"epsilon": 1e-6}),
        ]
    )
    intermediate_b = sequential_results[0]

    # Now create branch_b using the sequential output
    branch_b = ttnn.parallel.branch(ttnn.layer_norm, intermediate_b, cores=right_cores, epsilon=1e-6)

    # Execute parallel with both branches
    results = ttnn.parallel([branch_a, branch_b])

    # Verify branch A
    output_a = ttnn.to_torch(ttnn.from_device(results[0][0]))
    assert_with_pcc(torch_output_a, output_a, 0.999)

    # Verify branch B
    output_b = ttnn.to_torch(ttnn.from_device(results[1][0]))
    assert_with_pcc(torch_output_b, output_b, 0.999)

    print("✓ test_parallel_with_sequential_branch passed!")


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_deeply_nested_composition(device, batch_size, h, w):
    """
    Test: Deeply nested composition of parallel and sequential.

    Structure:
        ttnn.sequential([
            step_1: RMS norm,
            step_2: ttnn.parallel([
                branch_2a: LayerNorm,
                branch_2b: LayerNorm
            ]),
            step_3: another RMS norm on a different tensor
        ])

    This tests that we can arbitrarily compose these primitives.
    """
    torch.manual_seed(42)

    # Create test tensors
    torch_input1 = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_input2a = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_input2b = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_input3 = torch.rand((batch_size, h, w), dtype=torch.bfloat16)

    # Compute torch references
    torch_output1 = torch_rms_norm(torch_input1, eps=1e-5)
    torch_output2a = torch_layer_norm(torch_input2a, eps=1e-6)
    torch_output2b = torch_layer_norm(torch_input2b, eps=1e-6)
    torch_output3 = torch_rms_norm(torch_input3, eps=1e-7)

    # Define core ranges
    left_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])
    right_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 3))])

    # Move tensors to device
    input1 = ttnn.from_torch(torch_input1, device=device, layout=ttnn.TILE_LAYOUT)
    input2a = ttnn.from_torch(torch_input2a, device=device, layout=ttnn.TILE_LAYOUT)
    input2b = ttnn.from_torch(torch_input2b, device=device, layout=ttnn.TILE_LAYOUT)
    input3 = ttnn.from_torch(torch_input3, device=device, layout=ttnn.TILE_LAYOUT)

    # Create parallel branches for step 2
    branch_2a = ttnn.parallel.branch(ttnn.layer_norm, input2a, cores=left_cores, epsilon=1e-6)
    branch_2b = ttnn.parallel.branch(ttnn.layer_norm, input2b, cores=right_cores, epsilon=1e-6)

    # Execute the full sequential with nested parallel
    results = ttnn.sequential(
        [
            (ttnn.rms_norm, input1, {"epsilon": 1e-5}),  # Step 1
            (ttnn.parallel, [branch_2a, branch_2b]),  # Step 2 (parallel)
            (ttnn.rms_norm, input3, {"epsilon": 1e-7}),  # Step 3
        ]
    )

    # Verify Step 1
    output1 = ttnn.to_torch(ttnn.from_device(results[0]))
    assert_with_pcc(torch_output1, output1, 0.999)

    # Verify Step 2 (parallel results)
    parallel_results = results[1]
    output2a = ttnn.to_torch(ttnn.from_device(parallel_results[0][0]))
    output2b = ttnn.to_torch(ttnn.from_device(parallel_results[1][0]))
    assert_with_pcc(torch_output2a, output2a, 0.999)
    assert_with_pcc(torch_output2b, output2b, 0.999)

    # Verify Step 3
    output3 = ttnn.to_torch(ttnn.from_device(results[2]))
    assert_with_pcc(torch_output3, output3, 0.999)

    print("✓ test_deeply_nested_composition passed!")


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_chained_parallel_operations(device, batch_size, h, w):
    """
    Test: Chain multiple parallel operations using sequential.

    Structure:
        ttnn.sequential([
            step_1: ttnn.parallel([branch_1a, branch_1b]),
            step_2: ttnn.parallel([branch_2a, branch_2b]),
        ])

    This runs two parallel operations one after the other.
    """
    torch.manual_seed(42)

    # Create test tensors for first parallel
    torch_input1a = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_input1b = torch.rand((batch_size, h, w), dtype=torch.bfloat16)

    # Create test tensors for second parallel
    torch_input2a = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_input2b = torch.rand((batch_size, h, w), dtype=torch.bfloat16)

    # Compute torch references
    torch_output1a = torch_rms_norm(torch_input1a, eps=1e-5)
    torch_output1b = torch_rms_norm(torch_input1b, eps=1e-5)
    torch_output2a = torch_layer_norm(torch_input2a, eps=1e-6)
    torch_output2b = torch_layer_norm(torch_input2b, eps=1e-6)

    # Define core ranges
    left_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])
    right_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(7, 3))])

    # Move tensors to device
    input1a = ttnn.from_torch(torch_input1a, device=device, layout=ttnn.TILE_LAYOUT)
    input1b = ttnn.from_torch(torch_input1b, device=device, layout=ttnn.TILE_LAYOUT)
    input2a = ttnn.from_torch(torch_input2a, device=device, layout=ttnn.TILE_LAYOUT)
    input2b = ttnn.from_torch(torch_input2b, device=device, layout=ttnn.TILE_LAYOUT)

    # Create branches for first parallel (RMS norms)
    branch_1a = ttnn.parallel.branch(ttnn.rms_norm, input1a, cores=left_cores, epsilon=1e-5)
    branch_1b = ttnn.parallel.branch(ttnn.rms_norm, input1b, cores=right_cores, epsilon=1e-5)

    # Create branches for second parallel (LayerNorms)
    branch_2a = ttnn.parallel.branch(ttnn.layer_norm, input2a, cores=left_cores, epsilon=1e-6)
    branch_2b = ttnn.parallel.branch(ttnn.layer_norm, input2b, cores=right_cores, epsilon=1e-6)

    # Execute chained parallels
    results = ttnn.sequential(
        [
            (ttnn.parallel, [branch_1a, branch_1b]),  # First parallel
            (ttnn.parallel, [branch_2a, branch_2b]),  # Second parallel
        ]
    )

    # Verify first parallel results
    parallel1_results = results[0]
    output1a = ttnn.to_torch(ttnn.from_device(parallel1_results[0][0]))
    output1b = ttnn.to_torch(ttnn.from_device(parallel1_results[1][0]))
    assert_with_pcc(torch_output1a, output1a, 0.999)
    assert_with_pcc(torch_output1b, output1b, 0.999)

    # Verify second parallel results
    parallel2_results = results[1]
    output2a = ttnn.to_torch(ttnn.from_device(parallel2_results[0][0]))
    output2b = ttnn.to_torch(ttnn.from_device(parallel2_results[1][0]))
    assert_with_pcc(torch_output2a, output2a, 0.998)
    assert_with_pcc(torch_output2b, output2b, 0.998)

    print("✓ test_chained_parallel_operations passed!")
