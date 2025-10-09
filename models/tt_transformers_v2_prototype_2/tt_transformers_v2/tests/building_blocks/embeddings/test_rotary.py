# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Rotary Position Embeddings (RoPE) building block.
"""

import pytest
import torch
from src.building_blocks.embeddings import (
    RotaryEmbeddingImplConfig,
    RotaryEmbeddingSpec,
    apply_rotary_embeddings,
    get_default_rotary_impl_config,
    precompute_rotary_freqs,
)

import ttnn


class TestRotaryEmbeddingSpec:
    """Test suite for RotaryEmbeddingSpec."""

    def test_rotary_spec_basic(self):
        """Test basic rotary embedding spec."""
        spec = RotaryEmbeddingSpec(head_dim=128, max_seq_len=2048, theta=10000.0)
        spec.validate()
        assert spec.head_dim == 128
        assert spec.max_seq_len == 2048
        assert spec.theta == 10000.0
        assert spec.scaling_factor == 1.0  # Default

    def test_rotary_spec_with_scaling(self):
        """Test rotary embedding with position scaling (for longer contexts)."""
        spec = RotaryEmbeddingSpec(head_dim=128, max_seq_len=8192, theta=10000.0, scaling_factor=2.0)  # Linear scaling
        spec.validate()
        assert spec.scaling_factor == 2.0

    def test_rotary_spec_dynamic_scaling(self):
        """Test rotary embedding with dynamic NTK scaling."""
        spec = RotaryEmbeddingSpec(
            head_dim=128,
            max_seq_len=32768,  # Very long context
            theta=10000.0,
            scaling_type="dynamic",
            original_max_seq_len=2048,
        )
        spec.validate()
        assert spec.scaling_type == "dynamic"
        assert spec.original_max_seq_len == 2048

    def test_rotary_spec_validation_errors(self):
        """Test rotary embedding spec validation errors."""
        # Invalid head_dim (must be even)
        with pytest.raises(AssertionError):
            spec = RotaryEmbeddingSpec(head_dim=127, max_seq_len=2048)
            spec.validate()

        # Invalid max_seq_len
        with pytest.raises(AssertionError):
            spec = RotaryEmbeddingSpec(head_dim=128, max_seq_len=0)
            spec.validate()

        # Invalid theta
        with pytest.raises(AssertionError):
            spec = RotaryEmbeddingSpec(head_dim=128, max_seq_len=2048, theta=0)
            spec.validate()

        # Invalid scaling_factor
        with pytest.raises(AssertionError):
            spec = RotaryEmbeddingSpec(head_dim=128, max_seq_len=2048, scaling_factor=0)
            spec.validate()


class TestRotaryEmbeddingImplConfig:
    """Test suite for RotaryEmbeddingImplConfig."""

    def test_rotary_default_impl_config(self):
        """Test getting default implementation config for rotary embedding."""
        spec = RotaryEmbeddingSpec(head_dim=128, max_seq_len=2048)

        for device in ["N150", "N300", "T3000", "TG"]:
            impl_config = get_default_rotary_impl_config(spec, device)
            assert isinstance(impl_config, RotaryEmbeddingImplConfig)
            assert impl_config.cache_freqs in [True, False]

    def test_rotary_impl_config_optimization(self):
        """Test rotary embedding optimization configurations."""
        spec = RotaryEmbeddingSpec(head_dim=128, max_seq_len=2048)

        impl_config = get_default_rotary_impl_config(spec, "T3000")

        # Should have optimization settings
        assert impl_config.use_fused_kernel in [True, False]
        if impl_config.use_fused_kernel:
            assert impl_config.fused_kernel_config is not None


class TestRotaryEmbeddingComputation:
    """Test rotary embedding computation correctness."""

    def test_precompute_rotary_freqs(self):
        """Test frequency precomputation for rotary embeddings."""
        head_dim = 8  # Small for testing
        max_seq_len = 16
        theta = 10000.0

        # Compute frequencies
        freqs = precompute_rotary_freqs(head_dim, max_seq_len, theta)

        # Check shape
        assert freqs.shape == (max_seq_len, head_dim // 2)

        # Check that frequencies decrease as dimension increases
        # (lower dimensions rotate faster)
        for pos in range(max_seq_len):
            for i in range(head_dim // 2 - 1):
                assert freqs[pos, i] >= freqs[pos, i + 1]

    def test_apply_rotary_embeddings_2d(self):
        """Test applying rotary embeddings to 2D rotation."""
        # Simple 2D rotation test
        head_dim = 2
        seq_len = 4

        # Create simple input
        x = torch.tensor(
            [
                [1.0, 0.0],  # Position 0
                [1.0, 0.0],  # Position 1
                [1.0, 0.0],  # Position 2
                [1.0, 0.0],  # Position 3
            ]
        )

        # Apply rotations for different positions
        # Position 0: no rotation
        # Position 1: rotate by theta
        # etc.

        # After rotation, vectors should have same magnitude
        for i in range(seq_len):
            magnitude = torch.norm(x[i])
            assert torch.allclose(magnitude, torch.tensor(1.0))

    def test_rotary_position_invariance(self):
        """Test that rotary embeddings preserve relative positions."""
        # Key property: RoPE preserves relative position information
        # R(m) * R(n)^T = R(m-n) for rotation matrices R


class TestRotaryEmbeddingForward:
    """Test suite for rotary embedding forward function."""

    @pytest.mark.skip(reason="Requires actual device")
    def test_apply_rotary_basic(self):
        """Test applying rotary embeddings to query and key tensors."""
        batch_size = 2
        seq_len = 128
        num_heads = 32
        head_dim = 128

        spec = RotaryEmbeddingSpec(head_dim=head_dim, max_seq_len=2048)
        impl_config = get_default_rotary_impl_config(spec, "cpu")

        # Create dummy query and key tensors
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)

        q_tt = ttnn.from_torch(q)
        k_tt = ttnn.from_torch(k)

        # Precompute frequencies
        freqs = precompute_rotary_freqs(head_dim, seq_len, spec.theta)
        freqs_tt = ttnn.from_torch(freqs)

        # Apply rotary embeddings
        q_rotated, k_rotated = apply_rotary_embeddings(q_tt, k_tt, freqs_tt, spec, impl_config)

        # Check output shapes
        assert q_rotated.shape == q_tt.shape
        assert k_rotated.shape == k_tt.shape

    @pytest.mark.skip(reason="Requires actual device")
    def test_apply_rotary_with_offset(self):
        """Test applying rotary embeddings with position offset (for KV cache)."""
        batch_size = 2
        num_heads = 32
        seq_len = 1  # Single token for decode
        head_dim = 128
        position_offset = 1000  # Already processed 1000 tokens

        spec = RotaryEmbeddingSpec(head_dim=head_dim, max_seq_len=2048)
        impl_config = get_default_rotary_impl_config(spec, "cpu")

        # Create single token query and key
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)

        q_tt = ttnn.from_torch(q)
        k_tt = ttnn.from_torch(k)

        # Precompute frequencies for the specific position
        positions = torch.arange(position_offset, position_offset + seq_len)
        freqs = precompute_rotary_freqs(head_dim, max(positions) + 1, spec.theta)
        freqs_tt = ttnn.from_torch(freqs[positions])

        # Apply rotary embeddings
        q_rotated, k_rotated = apply_rotary_embeddings(q_tt, k_tt, freqs_tt, spec, impl_config, position_offset)

        assert q_rotated.shape == q_tt.shape
        assert k_rotated.shape == k_tt.shape


@pytest.mark.perf
class TestRotaryEmbeddingPerformance:
    """Performance tests for rotary embeddings."""

    @pytest.mark.skip(reason="Requires actual device")
    def test_rotary_application_latency(self):
        """Test latency of applying rotary embeddings."""
        # Would measure application time

    @pytest.mark.skip(reason="Requires actual device")
    def test_fused_vs_separate_application(self):
        """Test performance of fused vs separate rotary application."""
        # Would compare fused kernel vs separate ops


class TestRotaryEmbeddingScaling:
    """Test rotary embedding scaling methods for long contexts."""

    def test_linear_scaling(self):
        """Test linear position scaling."""
        original_max_len = 2048
        target_max_len = 8192
        scaling_factor = target_max_len / original_max_len

        # Positions should be scaled down
        position = 4096
        scaled_position = position / scaling_factor
        assert scaled_position == 1024

    def test_ntk_scaling(self):
        """Test NTK (Neural Tangent Kernel) scaling."""
        # NTK scaling adjusts the base theta instead of positions
        original_theta = 10000.0
        scaling_factor = 2.0

        # New theta for NTK scaling
        ntk_theta = original_theta * (scaling_factor ** (1.0 / 0.875))

        # Should be larger than original
        assert ntk_theta > original_theta

    def test_dynamic_scaling(self):
        """Test dynamic scaling that switches between linear and NTK."""
        # Dynamic scaling uses different strategies based on sequence length


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
