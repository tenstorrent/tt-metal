# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import os
import sys

sys.path.append(f'{os.environ["TT_METAL_HOME"]}/tt-train/sources/ttml')
import ttml  # noqa: E402


@pytest.fixture(autouse=True)
def reset_graph():
    """Reset the autograd graph before each test."""
    yield
    # Graph is automatically reset between tests


class TestLayerNormEpsilon:
    """Test epsilon parameter propagation through Python bindings."""

    def test_default_epsilon(self):
        """Test that default epsilon is used when not specified."""
        # Create test data
        features = 64
        input_data = np.ones((1, 1, 1, features), dtype=np.float32)

        # Create tensors using Tensor.from_numpy
        input_tensor = ttml.autograd.Tensor.from_numpy(input_data)
        gamma_tensor = ttml.autograd.Tensor.from_numpy(np.ones((1, 1, 1, features), dtype=np.float32))
        beta_tensor = ttml.autograd.Tensor.from_numpy(np.zeros((1, 1, 1, features), dtype=np.float32))

        # Call layernorm without eps parameter (should use default 1e-5F)
        output = ttml.ops.layernorm.layernorm(input_tensor, gamma_tensor, beta_tensor)

        # Verify output is valid (no NaN/Inf)
        output_np = output.to_numpy()
        assert not np.isnan(output_np).any(), "Default epsilon produced NaN"
        assert not np.isinf(output_np).any(), "Default epsilon produced Inf"

    def test_custom_epsilon_small(self):
        """Test that custom small epsilon value works correctly."""
        # Create test data
        features = 64
        custom_eps = 1e-12  # BERT standard
        input_data = np.ones((1, 1, 1, features), dtype=np.float32)

        # Create tensors
        input_tensor = ttml.autograd.Tensor.from_numpy(input_data)
        gamma_tensor = ttml.autograd.Tensor.from_numpy(np.ones((1, 1, 1, features), dtype=np.float32))
        beta_tensor = ttml.autograd.Tensor.from_numpy(np.zeros((1, 1, 1, features), dtype=np.float32))

        # Call layernorm with custom epsilon
        output = ttml.ops.layernorm.layernorm(input_tensor, gamma_tensor, beta_tensor, eps=custom_eps)

        # Verify output is valid (no NaN/Inf)
        output_np = output.to_numpy()
        assert not np.isnan(output_np).any(), "Custom epsilon produced NaN"
        assert not np.isinf(output_np).any(), "Custom epsilon produced Inf"

    def test_custom_epsilon_large(self):
        """Test that custom large epsilon value works correctly."""
        # Create test data
        features = 64
        custom_eps = 1e-2  # Larger epsilon
        input_data = np.ones((1, 1, 1, features), dtype=np.float32)

        # Create tensors
        input_tensor = ttml.autograd.Tensor.from_numpy(input_data)
        gamma_tensor = ttml.autograd.Tensor.from_numpy(np.ones((1, 1, 1, features), dtype=np.float32))
        beta_tensor = ttml.autograd.Tensor.from_numpy(np.zeros((1, 1, 1, features), dtype=np.float32))

        # Call layernorm with custom epsilon
        output = ttml.ops.layernorm.layernorm(input_tensor, gamma_tensor, beta_tensor, eps=custom_eps)

        # Verify output is valid (no NaN/Inf)
        output_np = output.to_numpy()
        assert not np.isnan(output_np).any(), "Large epsilon produced NaN"
        assert not np.isinf(output_np).any(), "Large epsilon produced Inf"

    def test_composite_layernorm_default_epsilon(self):
        """Test composite_layernorm with default epsilon."""
        # Create test data
        features = 64
        input_data = np.ones((1, 1, 1, features), dtype=np.float32)

        # Create tensors
        input_tensor = ttml.autograd.Tensor.from_numpy(input_data)
        gamma_tensor = ttml.autograd.Tensor.from_numpy(np.ones((1, 1, 1, features), dtype=np.float32))
        beta_tensor = ttml.autograd.Tensor.from_numpy(np.zeros((1, 1, 1, features), dtype=np.float32))

        # Call composite_layernorm without eps parameter
        output = ttml.ops.layernorm.composite_layernorm(input_tensor, gamma_tensor, beta_tensor)

        # Verify output is valid (no NaN/Inf)
        output_np = output.to_numpy()
        assert not np.isnan(output_np).any(), "Composite default epsilon produced NaN"
        assert not np.isinf(output_np).any(), "Composite default epsilon produced Inf"

    def test_composite_layernorm_custom_epsilon(self):
        """Test composite_layernorm with custom epsilon."""
        # Create test data
        features = 64
        custom_eps = 1e-6
        input_data = np.ones((1, 1, 1, features), dtype=np.float32)

        # Create tensors
        input_tensor = ttml.autograd.Tensor.from_numpy(input_data)
        gamma_tensor = ttml.autograd.Tensor.from_numpy(np.ones((1, 1, 1, features), dtype=np.float32))
        beta_tensor = ttml.autograd.Tensor.from_numpy(np.zeros((1, 1, 1, features), dtype=np.float32))

        # Call composite_layernorm with custom epsilon
        output = ttml.ops.layernorm.composite_layernorm(input_tensor, gamma_tensor, beta_tensor, eps=custom_eps)

        # Verify output is valid (no NaN/Inf)
        output_np = output.to_numpy()
        assert not np.isnan(output_np).any(), "Composite custom epsilon produced NaN"
        assert not np.isinf(output_np).any(), "Composite custom epsilon produced Inf"

    def test_zero_variance_with_epsilon(self):
        """Test epsilon prevents NaN with zero variance input."""
        # Create test data with zero variance
        features = 64
        custom_eps = 1e-5
        input_data = np.full((1, 1, 1, features), 42.0, dtype=np.float32)  # All same values

        # Create tensors
        input_tensor = ttml.autograd.Tensor.from_numpy(input_data)
        gamma_tensor = ttml.autograd.Tensor.from_numpy(np.ones((1, 1, 1, features), dtype=np.float32))
        beta_tensor = ttml.autograd.Tensor.from_numpy(np.zeros((1, 1, 1, features), dtype=np.float32))

        # Call layernorm with epsilon (should handle zero variance gracefully)
        output = ttml.ops.layernorm.layernorm(input_tensor, gamma_tensor, beta_tensor, eps=custom_eps)

        # Verify output is valid (no NaN/Inf)
        output_np = output.to_numpy()
        assert not np.isnan(output_np).any(), "Zero variance produced NaN despite epsilon"
        assert not np.isinf(output_np).any(), "Zero variance produced Inf despite epsilon"

        # With zero variance, normalized values should be near zero
        assert np.allclose(output_np, 0.0, atol=1e-3), "Zero variance should produce near-zero values"
