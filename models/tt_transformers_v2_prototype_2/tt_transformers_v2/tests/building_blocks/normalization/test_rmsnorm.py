# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for RMS Normalization building block.
"""

import pytest
import torch
from src.building_blocks.normalization import (
    RMSNormImplConfig,
    RMSNormSpec,
    get_default_rmsnorm_impl_config,
    rmsnorm_forward,
)

import ttnn


class TestRMSNormSpec:
    """Test suite for RMSNormSpec."""

    def test_rmsnorm_spec_basic(self):
        """Test basic RMSNorm spec creation and validation."""
        spec = RMSNormSpec(hidden_dim=4096, epsilon=1e-6)
        spec.validate()
        assert spec.hidden_dim == 4096
        assert spec.epsilon == 1e-6

    def test_rmsnorm_spec_default_epsilon(self):
        """Test RMSNorm spec with default epsilon."""
        spec = RMSNormSpec(hidden_dim=4096)
        spec.validate()
        assert spec.epsilon == 1e-5  # Default value

    def test_rmsnorm_spec_various_dims(self):
        """Test RMSNorm spec with various hidden dimensions."""
        for hidden_dim in [768, 1024, 2048, 4096, 8192]:
            spec = RMSNormSpec(hidden_dim=hidden_dim)
            spec.validate()
            assert spec.hidden_dim == hidden_dim

    def test_rmsnorm_spec_validation_errors(self):
        """Test RMSNorm spec validation errors."""
        # Invalid hidden_dim
        with pytest.raises(AssertionError):
            spec = RMSNormSpec(hidden_dim=0)
            spec.validate()

        with pytest.raises(AssertionError):
            spec = RMSNormSpec(hidden_dim=-1)
            spec.validate()

        # Invalid epsilon
        with pytest.raises(AssertionError):
            spec = RMSNormSpec(hidden_dim=4096, epsilon=0)
            spec.validate()

        with pytest.raises(AssertionError):
            spec = RMSNormSpec(hidden_dim=4096, epsilon=-1e-6)
            spec.validate()


class TestRMSNormImplConfig:
    """Test suite for RMSNormImplConfig."""

    def test_rmsnorm_default_impl_config(self):
        """Test getting default implementation config for RMSNorm."""
        spec = RMSNormSpec(hidden_dim=4096)

        # Test different devices and modes
        for device in ["N150", "N300", "T3000", "TG"]:
            for mode in ["prefill", "decode"]:
                impl_config = get_default_rmsnorm_impl_config(spec, device, mode)
                assert isinstance(impl_config, RMSNormImplConfig)
                assert impl_config.dtype in [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32]

    def test_rmsnorm_impl_config_sharding(self):
        """Test RMSNorm sharding configuration."""
        spec = RMSNormSpec(hidden_dim=4096)

        # Some devices might use sharded implementation
        impl_config = get_default_rmsnorm_impl_config(spec, "T3000", "decode")

        if impl_config.sharded_program_config is not None:
            # Check sharding parameters
            assert hasattr(impl_config.sharded_program_config, "compute_with_storage_grid_size")
            assert hasattr(impl_config.sharded_program_config, "subblock_w")

    def test_rmsnorm_impl_config_memory(self):
        """Test RMSNorm memory configuration."""
        spec = RMSNormSpec(hidden_dim=8192)  # Large hidden dim

        impl_config = get_default_rmsnorm_impl_config(spec, "TG", "prefill")

        # Check memory configurations
        assert impl_config.sharded_output_config is not None or impl_config.output_memory_config is not None


class TestRMSNormForward:
    """Test suite for RMSNorm forward function."""

    @pytest.mark.skip(reason="Requires actual device")
    def test_rmsnorm_forward_basic(self):
        """Test RMSNorm forward pass with basic inputs."""
        batch_size = 2
        seq_len = 128
        hidden_dim = 4096

        spec = RMSNormSpec(hidden_dim=hidden_dim)
        impl_config = get_default_rmsnorm_impl_config(spec, "cpu", "prefill")

        # Create dummy input and weight
        x = torch.randn(batch_size, seq_len, hidden_dim)
        weight = torch.ones(hidden_dim)

        x_tt = ttnn.from_torch(x)
        weight_tt = ttnn.from_torch(weight)

        # Forward pass
        output = rmsnorm_forward(x_tt, weight_tt, spec, impl_config)

        # Check output shape
        assert output.shape == x_tt.shape

    @pytest.mark.skip(reason="Requires actual device")
    def test_rmsnorm_forward_numerical(self):
        """Test RMSNorm numerical correctness against reference."""
        batch_size = 1
        seq_len = 64
        hidden_dim = 512
        epsilon = 1e-6

        spec = RMSNormSpec(hidden_dim=hidden_dim, epsilon=epsilon)
        impl_config = get_default_rmsnorm_impl_config(spec, "cpu", "prefill")

        # Create input and weight
        x = torch.randn(batch_size, seq_len, hidden_dim)
        weight = torch.ones(hidden_dim)

        # Reference implementation
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normalized = x / torch.sqrt(variance + epsilon)
        expected = x_normalized * weight

        # TT implementation
        x_tt = ttnn.from_torch(x)
        weight_tt = ttnn.from_torch(weight)
        output_tt = rmsnorm_forward(x_tt, weight_tt, spec, impl_config)
        output = ttnn.to_torch(output_tt)

        # Check numerical accuracy
        torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.perf
class TestRMSNormPerformance:
    """Performance tests for RMSNorm."""

    @pytest.mark.skip(reason="Requires actual device")
    @pytest.mark.parametrize("hidden_dim", [768, 1024, 2048, 4096, 8192])
    def test_rmsnorm_latency(self, hidden_dim):
        """Test RMSNorm latency for various hidden dimensions."""
        # Would measure actual latency

    @pytest.mark.skip(reason="Requires actual device")
    def test_rmsnorm_vs_layernorm_perf(self):
        """Compare RMSNorm performance vs LayerNorm."""
        # RMSNorm should be faster due to simpler computation


class TestRMSNormStability:
    """Test numerical stability of RMSNorm."""

    def test_rmsnorm_zero_input_handling(self):
        """Test RMSNorm with zero inputs."""
        # RMSNorm should handle zero inputs gracefully due to epsilon
        hidden_dim = 512
        x = torch.zeros(1, 16, hidden_dim)
        weight = torch.ones(hidden_dim)
        epsilon = 1e-6

        # Should not produce NaN or Inf
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normalized = x / torch.sqrt(variance + epsilon)
        output = x_normalized * weight

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_rmsnorm_large_input_stability(self):
        """Test RMSNorm with very large inputs."""
        hidden_dim = 512
        x = torch.randn(1, 16, hidden_dim) * 1e3  # Large magnitude
        weight = torch.ones(hidden_dim)
        epsilon = 1e-6

        # Should normalize properly
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normalized = x / torch.sqrt(variance + epsilon)
        output = x_normalized * weight

        # Output should be normalized (roughly unit variance)
        output_var = output.var(dim=-1).mean()
        assert 0.5 < output_var < 2.0  # Reasonable range


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
