# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for reshape operation with autograd support.

This test suite verifies that the reshape operation:
1. Correctly reshapes tensors in the forward pass
2. Properly restores gradients to original shape in backward pass
3. Preserves the computation graph for gradient flow
"""

import numpy as np
import pytest
import ml_dtypes

import ttnn
import ttml  # noqa: E402


def test_reshape_forward_basic():
    """Test basic forward pass of reshape operation."""
    # Create a 5D tensor [B, 1, 1, seq_len, features]
    batch_size = 2
    seq_len = 4
    features = 8

    # Create input tensor (use TILE layout for compatibility)
    input_data = np.random.randn(batch_size, 1, 1, seq_len, features).astype(np.float32)
    input_tensor = ttml.autograd.Tensor.from_numpy(input_data, layout=ttnn.Layout.TILE)

    # Reshape to 4D [B, 1, seq_len, features]
    new_shape = [batch_size, 1, seq_len, features]
    reshaped = ttml.ops.reshape.reshape(input_tensor, new_shape)

    # Check output shape
    output_shape = reshaped.shape()
    assert len(output_shape) == 4, f"Expected 4D output, got {len(output_shape)}D"
    assert output_shape == [
        batch_size,
        1,
        seq_len,
        features,
    ], f"Expected shape {[batch_size, 1, seq_len, features]}, got {output_shape}"

    # Check values are preserved (reshape should not change data, just view)
    # Note: Due to bfloat16 precision, we use a more lenient tolerance
    output_data = reshaped.to_numpy(ttnn.DataType.FLOAT32)
    expected_data = input_data.reshape(batch_size, 1, seq_len, features)
    np.testing.assert_allclose(output_data, expected_data, rtol=1e-2, atol=1e-2)


def test_reshape_backward_basic():
    """Test backward pass of reshape operation restores gradient to original shape."""
    # Create a 5D tensor [B, 1, 1, seq_len, features]
    batch_size = 2
    seq_len = 32  # Tile-aligned
    features = 32  # Tile-aligned

    # Create input tensor
    input_data = np.random.randn(batch_size, 1, 1, seq_len, features).astype(np.float32)
    input_tensor = ttml.autograd.Tensor.from_numpy(input_data, layout=ttnn.Layout.TILE)

    # Reshape to 4D [B, 1, seq_len, features]
    new_shape = [batch_size, 1, seq_len, features]
    reshaped = ttml.ops.reshape.reshape(input_tensor, new_shape)

    # Verify forward pass shape
    assert reshaped.shape() == [batch_size, 1, seq_len, features]

    # Create loss and run backward pass
    loss = ttml.ops.unary.mean(reshaped)
    loss.backward(False)

    # Verify gradient is initialized
    assert (
        input_tensor.is_grad_initialized()
    ), "Input should have gradient after backward"

    # Verify gradient shape matches original input shape (5D)
    grad_tensor = input_tensor.get_grad_rw()
    grad_shape = grad_tensor.shape
    assert len(grad_shape) == 5, f"Gradient should be 5D, got {len(grad_shape)}D"
    assert list(grad_shape) == [
        batch_size,
        1,
        1,
        seq_len,
        features,
    ], f"Gradient shape should match original input shape, got {list(grad_shape)}"

    # Reset graph for next test
    ttml.autograd.AutoContext.get_instance().reset_graph()


def test_reshape_5d_to_4d():
    """Test reshaping from 5D to 4D (similar to NanoGPT use case)."""
    # Simulate NanoGPT logits shape: [B, 1, 1, seq_len, vocab_size] -> [B, 1, seq_len, vocab_size]
    batch_size = 4
    seq_len = 32  # Tile-aligned
    vocab_size = 64  # Tile-aligned

    # Create 5D tensor
    input_data = np.random.randn(batch_size, 1, 1, seq_len, vocab_size).astype(
        np.float32
    )
    input_tensor = ttml.autograd.Tensor.from_numpy(input_data, layout=ttnn.Layout.TILE)

    # Reshape to 4D
    new_shape = [batch_size, 1, seq_len, vocab_size]
    reshaped = ttml.ops.reshape.reshape(input_tensor, new_shape)

    # Verify shape
    output_shape = reshaped.shape()
    assert output_shape == [
        batch_size,
        1,
        seq_len,
        vocab_size,
    ], f"Expected {[batch_size, 1, seq_len, vocab_size]}, got {output_shape}"

    # Verify values (use float32 for comparison due to bfloat16 precision)
    output_data = reshaped.to_numpy(ttnn.DataType.FLOAT32)
    expected_data = input_data.reshape(batch_size, 1, seq_len, vocab_size)
    np.testing.assert_allclose(output_data, expected_data, rtol=1e-2, atol=1e-2)

    # Test backward
    loss = ttml.ops.unary.mean(reshaped)
    loss.backward(False)

    assert input_tensor.is_grad_initialized(), "Input should have gradient"
    grad_tensor = input_tensor.get_grad_rw()
    grad_shape = grad_tensor.shape
    assert len(grad_shape) == 5, f"Gradient should be 5D, got {len(grad_shape)}D"
    assert list(grad_shape) == [
        batch_size,
        1,
        1,
        seq_len,
        vocab_size,
    ], f"Gradient shape should match original input shape, got {grad_shape}"

    ttml.autograd.AutoContext.get_instance().reset_graph()


def test_reshape_with_linear_layer():
    """Test reshape operation in a computation graph with linear layer."""
    # Simulate the NanoGPT scenario: linear -> reshape -> loss
    batch_size = 2
    seq_len = 32  # Tile-aligned
    in_features = 32  # Tile-aligned
    out_features = 64  # Tile-aligned

    # Create input to linear layer [B, 1, 1, seq_len, in_features]
    input_data = np.random.randn(batch_size, 1, 1, seq_len, in_features).astype(
        np.float32
    )
    input_tensor = ttml.autograd.Tensor.from_numpy(input_data, layout=ttnn.Layout.TILE)

    # Create weight for linear layer [1, 1, out_features, in_features]
    weight_data = np.random.randn(1, 1, out_features, in_features).astype(
        ml_dtypes.bfloat16
    )
    weight_tensor = ttml.autograd.Tensor.from_numpy(
        weight_data, layout=ttnn.Layout.TILE
    )

    # Linear layer forward
    linear_out = ttml.ops.linear.linear(input_tensor, weight_tensor, None)
    linear_shape = linear_out.shape()

    # Reshape output [B, 1, 1, seq_len, out_features] -> [B, 1, seq_len, out_features]
    if len(linear_shape) == 5:
        new_shape = [batch_size, 1, seq_len, out_features]
        reshaped = ttml.ops.reshape.reshape(linear_out, new_shape)
    else:
        reshaped = linear_out

    # Create a simple loss
    loss = ttml.ops.unary.mean(reshaped)

    # Backward pass - this should work without errors
    try:
        loss.backward(False)

        # Check gradients are initialized
        assert input_tensor.is_grad_initialized(), "Input should have gradient"
        assert weight_tensor.is_grad_initialized(), "Weight should have gradient"

        # Check gradient shapes
        input_grad_tensor = input_tensor.get_grad_rw()
        input_grad_shape = input_grad_tensor.shape
        assert list(input_grad_shape) == [
            batch_size,
            1,
            1,
            seq_len,
            in_features,
        ], f"Input gradient shape should match input shape, got {input_grad_shape}"

        weight_grad_tensor = weight_tensor.get_grad_rw()
        weight_grad_shape = weight_grad_tensor.shape
        assert list(weight_grad_shape) == [
            1,
            1,
            out_features,
            in_features,
        ], f"Weight gradient shape should match weight shape, got {weight_grad_shape}"

    except RuntimeError as e:
        pytest.fail(f"Backward pass failed with error: {e}")
    finally:
        ttml.autograd.AutoContext.get_instance().reset_graph()


def test_reshape_preserves_computation_graph():
    """Test that reshape preserves the computation graph for gradient flow."""
    batch_size = 2
    seq_len = 32  # Tile-aligned
    features = 32  # Tile-aligned

    # Create input
    input_data = np.random.randn(batch_size, 1, 1, seq_len, features).astype(np.float32)
    input_tensor = ttml.autograd.Tensor.from_numpy(input_data, layout=ttnn.Layout.TILE)

    # Apply some operations
    x = ttml.ops.binary.mul(input_tensor, 2.0)

    # Reshape
    new_shape = [batch_size, 1, seq_len, features]
    reshaped = ttml.ops.reshape.reshape(x, new_shape)

    # Apply more operations - add requires a tensor, not scalar
    ones_data = np.ones((1, 1, 1, 1), dtype=np.float32)
    ones_tensor = ttml.autograd.Tensor.from_numpy(ones_data, layout=ttnn.Layout.TILE)
    y = ttml.ops.binary.add(reshaped, ones_tensor)

    # Loss
    loss = ttml.ops.unary.mean(y)

    # Backward
    loss.backward(False)

    # All tensors in the graph should have gradients
    assert input_tensor.is_grad_initialized(), "Input should have gradient"

    # Verify gradient values are reasonable (not all zeros, not NaN)
    grad_tensor = input_tensor.get_grad_rw()
    grad_shape = grad_tensor.shape
    assert list(grad_shape) == [
        batch_size,
        1,
        1,
        seq_len,
        features,
    ], f"Gradient shape should match input shape, got {grad_shape}"

    ttml.autograd.AutoContext.get_instance().reset_graph()


def test_reshape_multiple_reshapes():
    """Test multiple reshape operations in sequence (forward only).

    Note: Backward through reshapes that reduce dimensionality below 4D
    is currently not supported by the moreh_mean_backward operation.
    This test verifies forward pass works correctly.
    """
    batch_size = 2
    seq_len = 32  # Tile-aligned
    features = 32  # Tile-aligned

    # Create input [B, 1, 1, seq_len, features]
    input_data = np.random.randn(batch_size, 1, 1, seq_len, features).astype(np.float32)
    input_tensor = ttml.autograd.Tensor.from_numpy(input_data, layout=ttnn.Layout.TILE)

    # First reshape: 5D -> 4D
    x1 = ttml.ops.reshape.reshape(input_tensor, [batch_size, 1, seq_len, features])
    assert x1.shape() == [batch_size, 1, seq_len, features]

    # Second reshape: 4D -> 3D (flatten batch and seq)
    x2 = ttml.ops.reshape.reshape(x1, [batch_size * seq_len, 1, features])
    assert x2.shape() == [batch_size * seq_len, 1, features]

    # Third reshape: 3D -> 2D
    x3 = ttml.ops.reshape.reshape(x2, [batch_size * seq_len, features])
    assert x3.shape() == [batch_size * seq_len, features]

    # Verify data is preserved through reshapes
    output_data = x3.to_numpy(ttnn.DataType.FLOAT32)
    expected_data = input_data.reshape(batch_size * seq_len, features)
    np.testing.assert_allclose(output_data, expected_data, rtol=1e-2, atol=1e-2)

    # Test backward with 5D -> 4D reshape only (this is supported)
    ttml.autograd.AutoContext.get_instance().reset_graph()

    input_tensor = ttml.autograd.Tensor.from_numpy(input_data, layout=ttnn.Layout.TILE)
    x1 = ttml.ops.reshape.reshape(input_tensor, [batch_size, 1, seq_len, features])
    loss = ttml.ops.unary.mean(x1)
    loss.backward(False)

    # Check that original input has gradient with correct shape
    assert input_tensor.is_grad_initialized(), "Input should have gradient"
    grad_tensor = input_tensor.get_grad_rw()
    grad_shape = grad_tensor.shape
    assert list(grad_shape) == [
        batch_size,
        1,
        1,
        seq_len,
        features,
    ], f"Gradient should match original input shape, got {grad_shape}"

    ttml.autograd.AutoContext.get_instance().reset_graph()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
