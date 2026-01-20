# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the ttnn.sequential C++ infrastructure.

These tests verify:
1. Step descriptor creation via the .step() API
2. Step descriptors as opaque objects
3. Step descriptors have the right attributes

Note: Full sequential-as-branch-in-parallel tests require operations that
properly respect the core range override. For now, those tests use the
Python-level sequential in test_parallel_sequential_composition.py.
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
def test_step_descriptor_creation_rms_norm(device, batch_size, h, w):
    """
    Test: Verify that rms_norm.step() creates a valid StepDescriptor.
    """
    torch.manual_seed(42)

    # Create test tensor
    torch_input = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)

    # Define cores for this step
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])

    # Create step descriptor with cores
    step = ttnn.rms_norm.step(input_tensor, cores, epsilon=1e-5)

    # Verify it's a StepDescriptor
    assert "StepDescriptor" in type(step).__name__, f"Expected StepDescriptor, got {type(step)}"

    print("✓ test_step_descriptor_creation_rms_norm passed!")


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_step_descriptor_creation_layer_norm(device, batch_size, h, w):
    """
    Test: Verify that layer_norm.step() creates a valid StepDescriptor.
    """
    torch.manual_seed(42)

    # Create test tensor
    torch_input = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)

    # Define cores for this step
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])

    # Create step descriptor with cores
    step = ttnn.layer_norm.step(input_tensor, cores, epsilon=1e-5)

    # Verify it's a StepDescriptor
    assert "StepDescriptor" in type(step).__name__, f"Expected StepDescriptor, got {type(step)}"

    print("✓ test_step_descriptor_creation_layer_norm passed!")


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_multiple_step_descriptors(device, batch_size, h, w):
    """
    Test: Create multiple step descriptors for different operations.
    """
    torch.manual_seed(42)

    # Create test tensors
    torch_input1 = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_input2 = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_input3 = torch.rand((batch_size, h, w), dtype=torch.bfloat16)

    input1 = ttnn.from_torch(torch_input1, device=device, layout=ttnn.TILE_LAYOUT)
    input2 = ttnn.from_torch(torch_input2, device=device, layout=ttnn.TILE_LAYOUT)
    input3 = ttnn.from_torch(torch_input3, device=device, layout=ttnn.TILE_LAYOUT)

    # Define cores for steps
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])

    # Create step descriptors with cores
    step1 = ttnn.rms_norm.step(input1, cores, epsilon=1e-5)
    step2 = ttnn.layer_norm.step(input2, cores, epsilon=1e-6)
    step3 = ttnn.rms_norm.step(input3, cores, epsilon=1e-7)

    # Verify all are StepDescriptors
    for i, step in enumerate([step1, step2, step3], 1):
        assert "StepDescriptor" in type(step).__name__, f"Step {i}: Expected StepDescriptor, got {type(step)}"

    # Verify they are distinct objects
    assert step1 is not step2
    assert step2 is not step3
    assert step1 is not step3

    print("✓ test_multiple_step_descriptors passed!")


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_step_with_weight(device, batch_size, h, w):
    """
    Test: Create step descriptors with weight tensors.
    """
    torch.manual_seed(42)

    # Create test tensors
    torch_input = torch.rand((batch_size, h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_input, device=device, layout=ttnn.TILE_LAYOUT)
    weight_tensor = ttnn.from_torch(torch_weight.reshape(1, 1, w), device=device, layout=ttnn.TILE_LAYOUT)

    # Define cores for this step
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))])

    # Create step descriptor with weight and cores
    step = ttnn.rms_norm.step(input_tensor, cores, epsilon=1e-5, weight=weight_tensor)

    # Verify it's a StepDescriptor
    assert "StepDescriptor" in type(step).__name__, f"Expected StepDescriptor, got {type(step)}"

    print("✓ test_step_with_weight passed!")


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_supports_sequential_property(device, batch_size, h, w):
    """
    Test: Verify the supports_sequential property on operations.
    """
    # Check rms_norm
    assert hasattr(ttnn.rms_norm, "supports_sequential"), "rms_norm should have supports_sequential property"
    assert ttnn.rms_norm.supports_sequential is True, "rms_norm should support sequential"

    # Check layer_norm
    assert hasattr(ttnn.layer_norm, "supports_sequential"), "layer_norm should have supports_sequential property"
    assert ttnn.layer_norm.supports_sequential is True, "layer_norm should support sequential"

    print("✓ test_supports_sequential_property passed!")
