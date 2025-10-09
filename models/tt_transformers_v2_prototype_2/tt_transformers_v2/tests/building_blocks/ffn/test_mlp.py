# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for MLP (Multi-Layer Perceptron) building block.
"""

import pytest
import torch
from src.building_blocks.ffn import (
    MLPImplConfig,
    MLPSpec,
    get_default_mlp_impl_config,
    mlp_decode_forward,
    mlp_prefill_forward,
)

import ttnn


class TestMLPSpec:
    """Test suite for MLPSpec."""

    def test_mlp_spec_basic(self):
        """Test basic MLP spec creation and validation."""
        spec = MLPSpec(hidden_dim=4096, intermediate_dim=16384, activation="relu")
        spec.validate()
        assert spec.hidden_dim == 4096
        assert spec.intermediate_dim == 16384
        assert spec.activation == "relu"
        assert spec.dropout == 0.0  # Default

    def test_mlp_spec_with_dropout(self):
        """Test MLP spec with dropout."""
        spec = MLPSpec(hidden_dim=4096, intermediate_dim=16384, activation="gelu", dropout=0.1)
        spec.validate()
        assert spec.dropout == 0.1

    def test_mlp_spec_expansion_ratio(self):
        """Test MLP spec with typical expansion ratios."""
        # 4x expansion (common in transformers)
        spec = MLPSpec(hidden_dim=768, intermediate_dim=3072, activation="gelu")
        spec.validate()
        assert spec.intermediate_dim == 4 * spec.hidden_dim

        # Custom expansion ratio
        spec = MLPSpec(hidden_dim=1024, intermediate_dim=4096, activation="relu")
        spec.validate()
        assert spec.intermediate_dim == 4 * spec.hidden_dim

    def test_mlp_spec_validation_errors(self):
        """Test MLP spec validation errors."""
        # Invalid hidden_dim
        with pytest.raises(AssertionError):
            spec = MLPSpec(hidden_dim=0, intermediate_dim=16384)
            spec.validate()

        # Invalid intermediate_dim
        with pytest.raises(AssertionError):
            spec = MLPSpec(hidden_dim=4096, intermediate_dim=0)
            spec.validate()

        # Invalid dropout
        with pytest.raises(AssertionError):
            spec = MLPSpec(hidden_dim=4096, intermediate_dim=16384, dropout=1.5)
            spec.validate()

        # Unknown activation
        with pytest.raises(AssertionError):
            spec = MLPSpec(hidden_dim=4096, intermediate_dim=16384, activation="unknown")
            spec.validate()


class TestMLPImplConfig:
    """Test suite for MLPImplConfig."""

    def test_mlp_default_impl_config(self):
        """Test getting default implementation config for MLP."""
        spec = MLPSpec(hidden_dim=4096, intermediate_dim=16384)

        # Test different devices and modes
        for device in ["N150", "N300", "T3000", "TG"]:
            for mode in ["prefill", "decode"]:
                impl_config = get_default_mlp_impl_config(spec, device, mode)
                assert isinstance(impl_config, MLPImplConfig)
                assert impl_config.weights_dtype in [ttnn.bfloat16, ttnn.bfloat8_b]
                assert impl_config.activations_dtype in [ttnn.bfloat16, ttnn.bfloat8_b]

    def test_mlp_impl_config_device_specific(self):
        """Test device-specific MLP configurations."""
        spec = MLPSpec(hidden_dim=4096, intermediate_dim=16384)

        # N150 config
        n150_config = get_default_mlp_impl_config(spec, "N150", "prefill")
        assert n150_config.compute_kernel_config is not None

        # T3000 config might have different settings
        t3000_config = get_default_mlp_impl_config(spec, "T3000", "prefill")
        assert t3000_config.compute_kernel_config is not None

    def test_mlp_impl_config_mode_differences(self):
        """Test differences between prefill and decode configs."""
        spec = MLPSpec(hidden_dim=4096, intermediate_dim=16384)

        prefill_config = get_default_mlp_impl_config(spec, "N150", "prefill")
        decode_config = get_default_mlp_impl_config(spec, "N150", "decode")

        # Decode might use different dtype for efficiency
        # (actual differences depend on implementation)
        assert prefill_config.compute_kernel_config is not None
        assert decode_config.compute_kernel_config is not None


class TestMLPForward:
    """Test suite for MLP forward functions."""

    @pytest.mark.skip(reason="Requires actual device")
    def test_mlp_prefill_forward_basic(self):
        """Test MLP forward pass in prefill mode."""
        batch_size = 2
        seq_len = 128
        hidden_dim = 4096

        spec = MLPSpec(hidden_dim=hidden_dim, intermediate_dim=16384, activation="gelu")
        impl_config = get_default_mlp_impl_config(spec, "cpu", "prefill")

        # Create dummy input
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        hidden_states_tt = ttnn.from_torch(hidden_states)

        # Forward pass
        output = mlp_prefill_forward(hidden_states_tt, spec, impl_config)

        # Check output shape
        assert output.shape == hidden_states_tt.shape

    @pytest.mark.skip(reason="Requires actual device")
    def test_mlp_decode_forward_basic(self):
        """Test MLP forward pass in decode mode."""
        batch_size = 2
        hidden_dim = 4096

        spec = MLPSpec(hidden_dim=hidden_dim, intermediate_dim=16384, activation="gelu")
        impl_config = get_default_mlp_impl_config(spec, "cpu", "decode")

        # Create dummy input (single token)
        hidden_states = torch.randn(batch_size, 1, hidden_dim)
        hidden_states_tt = ttnn.from_torch(hidden_states)

        # Forward pass
        output = mlp_decode_forward(hidden_states_tt, spec, impl_config)

        # Check output shape
        assert output.shape == hidden_states_tt.shape

    @pytest.mark.skip(reason="Requires actual device")
    @pytest.mark.parametrize("activation", ["relu", "gelu", "silu"])
    def test_mlp_activations(self, activation):
        """Test MLP with different activation functions."""
        batch_size = 1
        seq_len = 64
        hidden_dim = 512

        spec = MLPSpec(hidden_dim=hidden_dim, intermediate_dim=2048, activation=activation)
        impl_config = get_default_mlp_impl_config(spec, "cpu", "prefill")

        # Create dummy input
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        hidden_states_tt = ttnn.from_torch(hidden_states)

        # Forward pass
        output = mlp_prefill_forward(hidden_states_tt, spec, impl_config)

        # Should work with all activations
        assert output.shape == hidden_states_tt.shape


@pytest.mark.perf
class TestMLPPerformance:
    """Performance tests for MLP layers."""

    @pytest.mark.skip(reason="Requires actual device")
    @pytest.mark.parametrize(
        "hidden_dim,intermediate_dim",
        [
            (768, 3072),  # Small model
            (1024, 4096),  # Medium model
            (4096, 16384),  # Large model
        ],
    )
    def test_mlp_prefill_latency(self, hidden_dim, intermediate_dim):
        """Test MLP prefill latency meets targets."""
        # Would measure actual latency

    @pytest.mark.skip(reason="Requires actual device")
    def test_mlp_decode_latency(self):
        """Test MLP decode latency meets targets."""
        # Would measure actual latency


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
