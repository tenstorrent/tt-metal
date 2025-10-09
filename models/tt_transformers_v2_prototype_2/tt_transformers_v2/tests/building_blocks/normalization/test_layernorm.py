# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Layer Normalization building block.
"""

import pytest
import torch
from src.building_blocks.normalization import (
    LayerNormImplConfig,
    LayerNormSpec,
    get_default_layernorm_impl_config,
    layernorm_forward,
)

import ttnn


class TestLayerNormSpec:
    """Test suite for LayerNormSpec."""

    def test_layernorm_spec_basic(self):
        """Test basic LayerNorm spec creation and validation."""
        spec = LayerNormSpec(hidden_dim=4096, epsilon=1e-5, elementwise_affine=True)
        spec.validate()
        assert spec.hidden_dim == 4096
        assert spec.epsilon == 1e-5
        assert spec.elementwise_affine is True

    def test_layernorm_spec_no_affine(self):
        """Test LayerNorm spec without affine parameters."""
        spec = LayerNormSpec(hidden_dim=4096, elementwise_affine=False)
        spec.validate()
        assert spec.elementwise_affine is False

    def test_layernorm_spec_default_values(self):
        """Test LayerNorm spec with default values."""
        spec = LayerNormSpec(hidden_dim=4096)
        spec.validate()
        assert spec.epsilon == 1e-5  # Default
        assert spec.elementwise_affine is True  # Default

    def test_layernorm_spec_normalized_shape_variants(self):
        """Test LayerNorm with different normalized shapes."""
        # Last dimension normalization (most common)
        spec = LayerNormSpec(hidden_dim=768)
        spec.validate()
        assert spec.hidden_dim == 768

        # Could support multi-dimensional normalization in the future
        # e.g., normalized_shape=(64, 768) for normalizing last 2 dims

    def test_layernorm_spec_validation_errors(self):
        """Test LayerNorm spec validation errors."""
        # Invalid hidden_dim
        with pytest.raises(AssertionError):
            spec = LayerNormSpec(hidden_dim=0)
            spec.validate()

        with pytest.raises(AssertionError):
            spec = LayerNormSpec(hidden_dim=-1)
            spec.validate()

        # Invalid epsilon
        with pytest.raises(AssertionError):
            spec = LayerNormSpec(hidden_dim=4096, epsilon=0)
            spec.validate()

        with pytest.raises(AssertionError):
            spec = LayerNormSpec(hidden_dim=4096, epsilon=-1e-5)
            spec.validate()


class TestLayerNormImplConfig:
    """Test suite for LayerNormImplConfig."""

    def test_layernorm_default_impl_config(self):
        """Test getting default implementation config for LayerNorm."""
        spec = LayerNormSpec(hidden_dim=4096)

        # Test different devices and modes
        for device in ["N150", "N300", "T3000", "TG"]:
            for mode in ["prefill", "decode"]:
                impl_config = get_default_layernorm_impl_config(spec, device, mode)
                assert isinstance(impl_config, LayerNormImplConfig)
                assert impl_config.dtype in [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32]

    def test_layernorm_impl_config_compute_kernel(self):
        """Test LayerNorm compute kernel configuration."""
        spec = LayerNormSpec(hidden_dim=4096)

        impl_config = get_default_layernorm_impl_config(spec, "T3000", "prefill")

        # Should have compute kernel config
        assert impl_config.compute_kernel_config is not None
        if hasattr(impl_config.compute_kernel_config, "math_fidelity"):
            assert impl_config.compute_kernel_config.math_fidelity in [
                ttnn.MathFidelity.HiFi2,
                ttnn.MathFidelity.HiFi3,
                ttnn.MathFidelity.HiFi4,
            ]

    def test_layernorm_impl_config_memory(self):
        """Test LayerNorm memory configuration."""
        spec = LayerNormSpec(hidden_dim=8192)  # Large hidden dim

        impl_config = get_default_layernorm_impl_config(spec, "TG", "decode")

        # Should have memory configuration
        assert impl_config.output_memory_config is not None or impl_config.sharded_output_config is not None


class TestLayerNormForward:
    """Test suite for LayerNorm forward function."""

    @pytest.mark.skip(reason="Requires actual device")
    def test_layernorm_forward_basic(self):
        """Test LayerNorm forward pass with basic inputs."""
        batch_size = 2
        seq_len = 128
        hidden_dim = 4096

        spec = LayerNormSpec(hidden_dim=hidden_dim)
        impl_config = get_default_layernorm_impl_config(spec, "cpu", "prefill")

        # Create dummy input, weight, and bias
        x = torch.randn(batch_size, seq_len, hidden_dim)
        weight = torch.ones(hidden_dim)
        bias = torch.zeros(hidden_dim)

        x_tt = ttnn.from_torch(x)
        weight_tt = ttnn.from_torch(weight)
        bias_tt = ttnn.from_torch(bias)

        # Forward pass
        output = layernorm_forward(x_tt, weight_tt, spec, impl_config, bias=bias_tt)

        # Check output shape
        assert output.shape == x_tt.shape

    @pytest.mark.skip(reason="Requires actual device")
    def test_layernorm_forward_no_bias(self):
        """Test LayerNorm forward pass without bias."""
        batch_size = 2
        seq_len = 128
        hidden_dim = 4096

        spec = LayerNormSpec(hidden_dim=hidden_dim)
        impl_config = get_default_layernorm_impl_config(spec, "cpu", "prefill")

        # Create input and weight only (no bias)
        x = torch.randn(batch_size, seq_len, hidden_dim)
        weight = torch.ones(hidden_dim)

        x_tt = ttnn.from_torch(x)
        weight_tt = ttnn.from_torch(weight)

        # Forward pass without bias
        output = layernorm_forward(x_tt, weight_tt, spec, impl_config, bias=None)

        # Check output shape
        assert output.shape == x_tt.shape

    @pytest.mark.skip(reason="Requires actual device")
    def test_layernorm_forward_numerical(self):
        """Test LayerNorm numerical correctness against PyTorch reference."""
        batch_size = 1
        seq_len = 64
        hidden_dim = 512
        epsilon = 1e-5

        spec = LayerNormSpec(hidden_dim=hidden_dim, epsilon=epsilon)
        impl_config = get_default_layernorm_impl_config(spec, "cpu", "prefill")

        # Create input, weight, and bias
        x = torch.randn(batch_size, seq_len, hidden_dim)
        weight = torch.randn(hidden_dim)
        bias = torch.randn(hidden_dim)

        # PyTorch reference
        layer_norm = torch.nn.LayerNorm(hidden_dim, eps=epsilon)
        layer_norm.weight.data = weight
        layer_norm.bias.data = bias
        expected = layer_norm(x)

        # TT implementation
        x_tt = ttnn.from_torch(x)
        weight_tt = ttnn.from_torch(weight)
        bias_tt = ttnn.from_torch(bias)
        output_tt = layernorm_forward(x_tt, weight_tt, spec, impl_config, bias=bias_tt)
        output = ttnn.to_torch(output_tt)

        # Check numerical accuracy
        torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)


class TestLayerNormVsRMSNorm:
    """Compare LayerNorm with RMSNorm."""

    def test_computational_difference(self):
        """Test computational differences between LayerNorm and RMSNorm."""
        hidden_dim = 512
        x = torch.randn(1, 16, hidden_dim)
        weight = torch.ones(hidden_dim)
        epsilon = 1e-6

        # LayerNorm computation
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        ln_normalized = (x - mean) / torch.sqrt(var + epsilon)
        ln_output = ln_normalized * weight

        # RMSNorm computation (no mean subtraction)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + epsilon)
        rms_normalized = x / rms
        rms_output = rms_normalized * weight

        # Outputs should be different
        assert not torch.allclose(ln_output, rms_output)

        # But both should have similar scales
        assert torch.std(ln_output) > 0.5
        assert torch.std(rms_output) > 0.5


@pytest.mark.perf
class TestLayerNormPerformance:
    """Performance tests for LayerNorm."""

    @pytest.mark.skip(reason="Requires actual device")
    @pytest.mark.parametrize("hidden_dim", [768, 1024, 2048, 4096])
    def test_layernorm_latency(self, hidden_dim):
        """Test LayerNorm latency for various hidden dimensions."""
        # Would measure actual latency

    @pytest.mark.skip(reason="Requires actual device")
    def test_layernorm_memory_efficiency(self):
        """Test LayerNorm memory usage."""
        # Would measure memory consumption


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
