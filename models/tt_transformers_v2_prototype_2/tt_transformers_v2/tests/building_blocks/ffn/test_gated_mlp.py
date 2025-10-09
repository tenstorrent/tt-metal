# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Gated MLP building block (SwiGLU, GeGLU, etc.).
"""

import pytest
import torch
from src.building_blocks.ffn import (
    GatedMLPImplConfig,
    GatedMLPSpec,
    gated_mlp_decode_forward,
    gated_mlp_prefill_forward,
    get_default_gated_mlp_impl_config,
)

import ttnn


class TestGatedMLPSpec:
    """Test suite for GatedMLPSpec."""

    def test_gated_mlp_spec_swiglu(self):
        """Test SwiGLU (SiLU gated) MLP spec."""
        spec = GatedMLPSpec(
            hidden_dim=4096,
            intermediate_dim=11008,  # Common for Llama models
            activation="silu",
            gate_activation="silu",
        )
        spec.validate()
        assert spec.hidden_dim == 4096
        assert spec.intermediate_dim == 11008
        assert spec.activation == "silu"
        assert spec.gate_activation == "silu"

    def test_gated_mlp_spec_geglu(self):
        """Test GeGLU (GELU gated) MLP spec."""
        spec = GatedMLPSpec(hidden_dim=4096, intermediate_dim=16384, activation="gelu", gate_activation="gelu")
        spec.validate()
        assert spec.activation == "gelu"
        assert spec.gate_activation == "gelu"

    def test_gated_mlp_spec_mixed_activations(self):
        """Test gated MLP with different gate and activation functions."""
        spec = GatedMLPSpec(hidden_dim=4096, intermediate_dim=11008, activation="relu", gate_activation="sigmoid")
        spec.validate()
        assert spec.activation == "relu"
        assert spec.gate_activation == "sigmoid"

    def test_gated_mlp_spec_validation_errors(self):
        """Test gated MLP spec validation errors."""
        # Invalid dimensions
        with pytest.raises(AssertionError):
            spec = GatedMLPSpec(hidden_dim=0, intermediate_dim=11008, activation="silu")
            spec.validate()

        # Unknown activation
        with pytest.raises(AssertionError):
            spec = GatedMLPSpec(hidden_dim=4096, intermediate_dim=11008, activation="unknown")
            spec.validate()

        # Unknown gate activation
        with pytest.raises(AssertionError):
            spec = GatedMLPSpec(hidden_dim=4096, intermediate_dim=11008, activation="silu", gate_activation="unknown")
            spec.validate()


class TestGatedMLPImplConfig:
    """Test suite for GatedMLPImplConfig."""

    def test_gated_mlp_default_impl_config(self):
        """Test getting default implementation config for gated MLP."""
        spec = GatedMLPSpec(hidden_dim=4096, intermediate_dim=11008, activation="silu")

        # Test different devices
        for device in ["N150", "N300", "T3000", "TG"]:
            for mode in ["prefill", "decode"]:
                impl_config = get_default_gated_mlp_impl_config(spec, device, mode)
                assert isinstance(impl_config, GatedMLPImplConfig)
                assert impl_config.fuse_gate_and_up_proj in [True, False]

    def test_gated_mlp_impl_config_fusion(self):
        """Test gated MLP fusion configuration."""
        spec = GatedMLPSpec(hidden_dim=4096, intermediate_dim=11008, activation="silu")

        # Some devices might support fused gate operations
        impl_config = get_default_gated_mlp_impl_config(spec, "T3000", "prefill")

        if impl_config.fuse_gate_and_up_proj:
            # Fused implementation should have specific config
            assert impl_config.fused_kernel_config is not None
        else:
            # Non-fused uses separate kernels
            assert impl_config.gate_kernel_config is not None
            assert impl_config.up_proj_kernel_config is not None


class TestGatedMLPForward:
    """Test suite for gated MLP forward functions."""

    @pytest.mark.skip(reason="Requires actual device")
    def test_gated_mlp_prefill_forward_basic(self):
        """Test gated MLP forward pass in prefill mode."""
        batch_size = 2
        seq_len = 128
        hidden_dim = 4096

        spec = GatedMLPSpec(hidden_dim=hidden_dim, intermediate_dim=11008, activation="silu", gate_activation="silu")
        impl_config = get_default_gated_mlp_impl_config(spec, "cpu", "prefill")

        # Create dummy input
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        hidden_states_tt = ttnn.from_torch(hidden_states)

        # Forward pass
        output = gated_mlp_prefill_forward(hidden_states_tt, spec, impl_config)

        # Check output shape
        assert output.shape == hidden_states_tt.shape

    @pytest.mark.skip(reason="Requires actual device")
    def test_gated_mlp_decode_forward_basic(self):
        """Test gated MLP forward pass in decode mode."""
        batch_size = 2
        hidden_dim = 4096

        spec = GatedMLPSpec(hidden_dim=hidden_dim, intermediate_dim=11008, activation="silu")
        impl_config = get_default_gated_mlp_impl_config(spec, "cpu", "decode")

        # Create dummy input (single token)
        hidden_states = torch.randn(batch_size, 1, hidden_dim)
        hidden_states_tt = ttnn.from_torch(hidden_states)

        # Forward pass
        output = gated_mlp_decode_forward(hidden_states_tt, spec, impl_config)

        # Check output shape
        assert output.shape == hidden_states_tt.shape


class TestGatedMLPNumerics:
    """Test numerical behavior of gated MLP."""

    def test_swiglu_computation(self):
        """Test SwiGLU computation against reference."""
        # This would test the actual gating mechanism
        # gate(x) * up(x) where gate uses SiLU activation

    def test_geglu_computation(self):
        """Test GeGLU computation against reference."""
        # This would test GELU gating


@pytest.mark.perf
class TestGatedMLPPerformance:
    """Performance tests for gated MLP layers."""

    @pytest.mark.skip(reason="Requires actual device")
    def test_gated_mlp_vs_standard_mlp(self):
        """Compare gated MLP performance vs standard MLP."""
        # Would measure if gating overhead is acceptable

    @pytest.mark.skip(reason="Requires actual device")
    def test_fused_vs_separate_gates(self):
        """Test performance of fused vs separate gate operations."""
        # Would compare fused and non-fused implementations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
