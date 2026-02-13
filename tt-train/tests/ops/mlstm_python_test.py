# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Python mLSTM implementation.

Tests cover:
- mlstm_parallel: Core parallel computation
- mLSTMCell: Cell with gates
- mLSTMLayer: Full layer with projections
- mLSTMBlock: Pre-normed block
- xLSTMStack: Stack of blocks
"""

import numpy as np
import pytest


def mlstm_parallel_reference_numpy(
    matQ: np.ndarray,
    matK: np.ndarray,
    matV: np.ndarray,
    vecI: np.ndarray,
    vecF: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Reference mLSTM forward pass implementation in NumPy.

    This is a direct translation of the JAX implementation from mlstm_kernels.
    """
    B, NH, S, DHQK = matQ.shape

    # Step 1: log_sigmoid of forget gate
    vecLogSigF = np.log(1.0 / (1.0 + np.exp(-vecF)))

    # Step 2: Cumulative sum
    vecLogSigF_cumsum = np.cumsum(vecLogSigF, axis=2)

    # Step 3: Create log forget gate matrix
    matLogSigF = (
        vecLogSigF_cumsum[:, :, :, np.newaxis] - vecLogSigF_cumsum[:, :, np.newaxis, :]
    )

    # Step 4: Apply lower triangular mask
    ltr = np.tril(np.ones((S, S), dtype=np.bool_))
    matLogSigF_mask = np.where(ltr, matLogSigF, -np.inf)

    # Step 5: Add input gate
    matLogD = matLogSigF_mask + vecI[:, :, np.newaxis, :]

    # Step 6: Row-wise max for stabilization
    vecM = np.max(matLogD, axis=-1, keepdims=True)

    # Step 7: Stabilized D matrix
    matLogD_stabilized = matLogD - vecM
    matD = np.exp(matLogD_stabilized)

    # Step 8: Scaled dot product
    scale = DHQK**-0.5
    matS = (matQ @ matK.swapaxes(-2, -1)) * scale

    # Step 9: Gated attention
    matCtilde = matS * matD

    # Step 10: Normalizer
    vecN = np.maximum(np.abs(np.sum(matCtilde, axis=-1, keepdims=True)), np.exp(-vecM))

    # Step 11: Normalize
    matC = matCtilde / (vecN + eps)

    # Step 12: Output
    matH = matC @ matV

    return matH


def test_mlstm_forward_basic():
    """Test basic mLSTM forward pass."""
    import ttml
    from ttml.ops import mlstm_parallel

    # Initialize device
    auto_ctx = ttml.autograd.AutoContext.get_instance()
    auto_ctx.open_device()

    try:
        # Create random inputs
        B, NH, S, D = 1, 2, 32, 32
        np.random.seed(42)

        query_data = np.random.randn(B, NH, S, D).astype(np.float32) * 0.5
        key_data = np.random.randn(B, NH, S, D).astype(np.float32) * 0.5
        value_data = np.random.randn(B, NH, S, D).astype(np.float32) * 0.5
        input_gate_data = np.random.randn(B, NH, S).astype(np.float32)
        forget_gate_data = (
            np.random.randn(B, NH, S).astype(np.float32) + 2.0
        )  # Bias towards keeping

        # Create tensors
        query = ttml.autograd.Tensor.from_numpy(query_data)
        key = ttml.autograd.Tensor.from_numpy(key_data)
        value = ttml.autograd.Tensor.from_numpy(value_data)
        input_gate = ttml.autograd.Tensor.from_numpy(input_gate_data)
        forget_gate = ttml.autograd.Tensor.from_numpy(forget_gate_data)

        # Run forward pass
        output = mlstm_parallel(query, key, value, input_gate, forget_gate)

        # Get result
        result = output.get_value().to_numpy()

        # Compute reference
        expected = mlstm_parallel_reference_numpy(
            query_data, key_data, value_data, input_gate_data, forget_gate_data
        )

        # Check shapes match
        assert (
            result.shape == expected.shape
        ), f"Shape mismatch: {result.shape} vs {expected.shape}"

        # Check values are close (with tolerance for numerical precision)
        mse = np.mean((result - expected) ** 2)
        print(f"MSE between result and reference: {mse}")
        assert mse < 0.1, f"MSE too high: {mse}"

    finally:
        auto_ctx.close_device()


def test_mlstm_backward_gradient_flow():
    """Test that gradients flow correctly through mLSTM."""
    import ttml
    from ttml.ops import mlstm_parallel

    # Initialize device
    auto_ctx = ttml.autograd.AutoContext.get_instance()
    auto_ctx.open_device()

    try:
        # Create small inputs for gradient testing
        B, NH, S, D = 1, 1, 32, 32
        np.random.seed(42)

        query_data = np.random.randn(B, NH, S, D).astype(np.float32) * 0.5
        key_data = np.random.randn(B, NH, S, D).astype(np.float32) * 0.5
        value_data = np.random.randn(B, NH, S, D).astype(np.float32) * 0.5
        input_gate_data = np.random.randn(B, NH, S).astype(np.float32)
        forget_gate_data = np.random.randn(B, NH, S).astype(np.float32) + 2.0

        # Create tensors with requires_grad=True
        query = ttml.autograd.Tensor.from_numpy(query_data)
        key = ttml.autograd.Tensor.from_numpy(key_data)
        value = ttml.autograd.Tensor.from_numpy(value_data)
        input_gate = ttml.autograd.Tensor.from_numpy(input_gate_data)
        forget_gate = ttml.autograd.Tensor.from_numpy(forget_gate_data)

        # Forward pass
        output = mlstm_parallel(query, key, value, input_gate, forget_gate)

        # Backward pass
        output.backward()

        # Check gradients are initialized
        assert query.is_grad_initialized(), "Query gradient not initialized"
        assert key.is_grad_initialized(), "Key gradient not initialized"
        assert value.is_grad_initialized(), "Value gradient not initialized"
        assert input_gate.is_grad_initialized(), "Input gate gradient not initialized"
        assert forget_gate.is_grad_initialized(), "Forget gate gradient not initialized"

        # Check gradient shapes
        assert (
            query.get_grad().shape == query.get_value().shape
        ), "Query gradient shape mismatch"
        assert (
            key.get_grad().shape == key.get_value().shape
        ), "Key gradient shape mismatch"
        assert (
            value.get_grad().shape == value.get_value().shape
        ), "Value gradient shape mismatch"
        assert (
            input_gate.get_grad().shape == input_gate.get_value().shape
        ), "Input gate gradient shape mismatch"
        assert (
            forget_gate.get_grad().shape == forget_gate.get_value().shape
        ), "Forget gate gradient shape mismatch"

        # Check gradients are non-zero
        q_grad = query.get_grad().to_numpy()
        k_grad = key.get_grad().to_numpy()
        v_grad = value.get_grad().to_numpy()

        assert np.sum(np.abs(q_grad)) > 0, "Query gradient is all zeros"
        assert np.sum(np.abs(k_grad)) > 0, "Key gradient is all zeros"
        assert np.sum(np.abs(v_grad)) > 0, "Value gradient is all zeros"

        print("All gradient tests passed!")

    finally:
        auto_ctx.close_device()


def test_mlstm_cell_forward():
    """Test mLSTMCell forward pass."""
    import ttml
    from ttml.ops import mLSTMCell, mLSTMCellConfig

    # Initialize device
    auto_ctx = ttml.autograd.AutoContext.get_instance()
    auto_ctx.open_device()

    try:
        # Create cell
        config = mLSTMCellConfig(
            context_length=32,
            embedding_dim=64,
            num_heads=4,
        )
        cell = mLSTMCell(config)

        # Create inputs
        B, S, H = 1, 32, 64
        np.random.seed(42)

        q_data = np.random.randn(B, S, H).astype(np.float32) * 0.5
        k_data = np.random.randn(B, S, H).astype(np.float32) * 0.5
        v_data = np.random.randn(B, S, H).astype(np.float32) * 0.5

        q = ttml.autograd.Tensor.from_numpy(q_data)
        k = ttml.autograd.Tensor.from_numpy(k_data)
        v = ttml.autograd.Tensor.from_numpy(v_data)

        # Forward pass
        output = cell(q, k, v)

        # Check output shape
        assert output.get_value().shape == (
            B,
            S,
            H,
        ), f"Expected shape {(B, S, H)}, got {output.get_value().shape}"

        print("mLSTMCell forward test passed!")

    finally:
        auto_ctx.close_device()


def test_mlstm_layer_forward():
    """Test mLSTMLayer forward pass."""
    import ttml
    from ttml.ops import mLSTMLayer, mLSTMLayerConfig

    # Initialize device
    auto_ctx = ttml.autograd.AutoContext.get_instance()
    auto_ctx.open_device()

    try:
        # Create layer
        config = mLSTMLayerConfig(
            embedding_dim=64,
            num_heads=4,
            proj_factor=2.0,
            conv1d_kernel_size=4,
            context_length=32,
        )
        layer = mLSTMLayer(config)

        # Create input
        B, S, H = 1, 32, 64
        np.random.seed(42)
        x_data = np.random.randn(B, S, H).astype(np.float32) * 0.5
        x = ttml.autograd.Tensor.from_numpy(x_data)

        # Forward pass
        output = layer(x)

        # Check output shape matches input
        assert output.get_value().shape == (
            B,
            S,
            H,
        ), f"Expected shape {(B, S, H)}, got {output.get_value().shape}"

        print("mLSTMLayer forward test passed!")

    finally:
        auto_ctx.close_device()


def test_mlstm_block_forward():
    """Test mLSTMBlock forward pass."""
    import ttml
    from ttml.ops import mLSTMBlock, mLSTMBlockConfig, mLSTMLayerConfig

    # Initialize device
    auto_ctx = ttml.autograd.AutoContext.get_instance()
    auto_ctx.open_device()

    try:
        # Create block
        layer_config = mLSTMLayerConfig(
            embedding_dim=64,
            num_heads=4,
            proj_factor=2.0,
            context_length=32,
        )
        config = mLSTMBlockConfig(mlstm=layer_config)
        block = mLSTMBlock(config)

        # Create input
        B, S, H = 1, 32, 64
        np.random.seed(42)
        x_data = np.random.randn(B, S, H).astype(np.float32) * 0.5
        x = ttml.autograd.Tensor.from_numpy(x_data)

        # Forward pass
        output = block(x)

        # Check output shape matches input (residual connection)
        assert output.get_value().shape == (
            B,
            S,
            H,
        ), f"Expected shape {(B, S, H)}, got {output.get_value().shape}"

        print("mLSTMBlock forward test passed!")

    finally:
        auto_ctx.close_device()


def test_xlstm_stack_forward():
    """Test xLSTMStack forward pass."""
    import ttml
    from ttml.ops import xLSTMStack

    # Initialize device
    auto_ctx = ttml.autograd.AutoContext.get_instance()
    auto_ctx.open_device()

    try:
        # Create stack with 2 blocks
        stack = xLSTMStack(
            embedding_dim=64,
            num_blocks=2,
            num_heads=4,
            proj_factor=2.0,
            context_length=32,
        )

        # Create input
        B, S, H = 1, 32, 64
        np.random.seed(42)
        x_data = np.random.randn(B, S, H).astype(np.float32) * 0.5
        x = ttml.autograd.Tensor.from_numpy(x_data)

        # Forward pass
        output = stack(x)

        # Check output shape matches input
        assert output.get_value().shape == (
            B,
            S,
            H,
        ), f"Expected shape {(B, S, H)}, got {output.get_value().shape}"

        print("xLSTMStack forward test passed!")

    finally:
        auto_ctx.close_device()


if __name__ == "__main__":
    print("Running mLSTM Python tests...")

    print("\n1. Testing mlstm_parallel forward...")
    test_mlstm_forward_basic()
    print("   PASSED")

    print("\n2. Testing mlstm_parallel backward...")
    test_mlstm_backward_gradient_flow()
    print("   PASSED")

    print("\n3. Testing mLSTMCell...")
    test_mlstm_cell_forward()
    print("   PASSED")

    print("\n4. Testing mLSTMLayer...")
    test_mlstm_layer_forward()
    print("   PASSED")

    print("\n5. Testing mLSTMBlock...")
    test_mlstm_block_forward()
    print("   PASSED")

    print("\n6. Testing xLSTMStack...")
    test_xlstm_stack_forward()
    print("   PASSED")

    print("\n" + "=" * 50)
    print("All mLSTM tests passed!")
    print("=" * 50)
