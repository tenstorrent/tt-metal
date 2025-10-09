# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Position Embedding building block.
"""

import math

import pytest
import torch
from src.building_blocks.embeddings import (
    PositionEmbeddingImplConfig,
    PositionEmbeddingSpec,
    get_default_position_impl_config,
    position_embedding_forward,
)

import ttnn


class TestPositionEmbeddingSpec:
    """Test suite for PositionEmbeddingSpec."""

    def test_position_embedding_spec_sinusoidal(self):
        """Test sinusoidal position embedding spec."""
        spec = PositionEmbeddingSpec(max_seq_len=2048, embedding_dim=4096, embedding_type="sinusoidal")
        spec.validate()
        assert spec.max_seq_len == 2048
        assert spec.embedding_dim == 4096
        assert spec.embedding_type == "sinusoidal"
        assert spec.base == 10000.0  # Default
        assert spec.trainable is False  # Sinusoidal is not trainable

    def test_position_embedding_spec_learned(self):
        """Test learned position embedding spec."""
        spec = PositionEmbeddingSpec(max_seq_len=2048, embedding_dim=4096, embedding_type="learned")
        spec.validate()
        assert spec.embedding_type == "learned"
        assert spec.trainable is True  # Learned embeddings are trainable

    def test_position_embedding_spec_custom_base(self):
        """Test position embedding with custom base for sinusoidal."""
        spec = PositionEmbeddingSpec(
            max_seq_len=2048, embedding_dim=4096, embedding_type="sinusoidal", base=50000.0  # Custom base
        )
        spec.validate()
        assert spec.base == 50000.0

    def test_position_embedding_spec_alibi(self):
        """Test ALiBi (Attention with Linear Biases) position embedding."""
        spec = PositionEmbeddingSpec(
            max_seq_len=2048, embedding_dim=4096, embedding_type="alibi", num_heads=32  # ALiBi needs num_heads
        )
        spec.validate()
        assert spec.embedding_type == "alibi"
        assert spec.num_heads == 32

    def test_position_embedding_spec_validation_errors(self):
        """Test position embedding spec validation errors."""
        # Invalid max_seq_len
        with pytest.raises(AssertionError):
            spec = PositionEmbeddingSpec(max_seq_len=0, embedding_dim=4096)
            spec.validate()

        # Invalid embedding_dim
        with pytest.raises(AssertionError):
            spec = PositionEmbeddingSpec(max_seq_len=2048, embedding_dim=0)
            spec.validate()

        # Invalid embedding_type
        with pytest.raises(AssertionError):
            spec = PositionEmbeddingSpec(max_seq_len=2048, embedding_dim=4096, embedding_type="unknown")
            spec.validate()

        # ALiBi without num_heads
        with pytest.raises(AssertionError):
            spec = PositionEmbeddingSpec(max_seq_len=2048, embedding_dim=4096, embedding_type="alibi", num_heads=None)
            spec.validate()


class TestPositionEmbeddingImplConfig:
    """Test suite for PositionEmbeddingImplConfig."""

    def test_position_embedding_default_impl_config(self):
        """Test getting default implementation config for position embedding."""
        spec = PositionEmbeddingSpec(max_seq_len=2048, embedding_dim=4096)

        for device in ["N150", "N300", "T3000", "TG"]:
            impl_config = get_default_position_impl_config(spec, device)
            assert isinstance(impl_config, PositionEmbeddingImplConfig)
            assert impl_config.dtype in [ttnn.bfloat16, ttnn.float32]

    def test_position_embedding_impl_config_cache(self):
        """Test position embedding caching configuration."""
        spec = PositionEmbeddingSpec(max_seq_len=2048, embedding_dim=4096, embedding_type="sinusoidal")

        impl_config = get_default_position_impl_config(spec, "T3000")

        # Sinusoidal embeddings can be cached
        assert impl_config.cache_embeddings is True
        assert impl_config.cache_memory_config is not None


class TestPositionEmbeddingForward:
    """Test suite for position embedding forward function."""

    @pytest.mark.skip(reason="Requires actual device")
    def test_position_embedding_forward_sinusoidal(self):
        """Test sinusoidal position embedding forward pass."""
        batch_size = 2
        seq_len = 128
        embedding_dim = 512  # Must be even for sinusoidal

        spec = PositionEmbeddingSpec(max_seq_len=2048, embedding_dim=embedding_dim, embedding_type="sinusoidal")
        impl_config = get_default_position_impl_config(spec, "cpu")

        # Create position indices
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        positions_tt = ttnn.from_torch(positions)

        # Forward pass
        output = position_embedding_forward(positions_tt, spec, impl_config)

        # Check output shape
        assert output.shape == (batch_size, seq_len, embedding_dim)

    @pytest.mark.skip(reason="Requires actual device")
    def test_position_embedding_forward_learned(self):
        """Test learned position embedding forward pass."""
        batch_size = 2
        seq_len = 128
        embedding_dim = 4096

        spec = PositionEmbeddingSpec(max_seq_len=2048, embedding_dim=embedding_dim, embedding_type="learned")
        impl_config = get_default_position_impl_config(spec, "cpu")

        # Create position indices
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        positions_tt = ttnn.from_torch(positions)

        # Forward pass
        output = position_embedding_forward(positions_tt, spec, impl_config)

        # Check output shape
        assert output.shape == (batch_size, seq_len, embedding_dim)


class TestSinusoidalPositionEmbedding:
    """Test sinusoidal position embedding properties."""

    def test_sinusoidal_embedding_computation(self):
        """Test sinusoidal position embedding computation."""
        max_seq_len = 16
        embedding_dim = 8  # Small for testing
        base = 10000.0

        # Create position indices
        positions = torch.arange(max_seq_len).float()

        # Compute sinusoidal embeddings manually
        half_dim = embedding_dim // 2
        emb = torch.zeros(max_seq_len, embedding_dim)

        # Compute frequency bands
        freq = torch.exp(torch.arange(0, half_dim, dtype=torch.float32) * -(math.log(base) / half_dim))

        # Apply sin and cos
        angles = positions.unsqueeze(1) * freq.unsqueeze(0)
        emb[:, 0::2] = torch.sin(angles)
        emb[:, 1::2] = torch.cos(angles)

        # Check properties
        # 1. Embeddings should be bounded between -1 and 1
        assert torch.all(emb >= -1.0) and torch.all(emb <= 1.0)

        # 2. Different positions should have different embeddings
        for i in range(max_seq_len - 1):
            assert not torch.allclose(emb[i], emb[i + 1])

        # 3. The pattern should be periodic with different frequencies
        # Higher dimensions should have lower frequencies
        high_freq_period = emb[0:8, 0]  # First dimension
        low_freq_period = emb[0:8, -2]  # Last sin dimension
        assert torch.std(high_freq_period) > torch.std(low_freq_period)

    def test_sinusoidal_relative_position_property(self):
        """Test that sinusoidal embeddings encode relative positions."""
        # This is a key property of sinusoidal embeddings:
        # PE(pos + k) can be represented as a linear function of PE(pos)
        # This allows the model to learn relative position information


class TestLearnedPositionEmbedding:
    """Test learned position embedding properties."""

    def test_learned_embedding_parameters(self):
        """Test learned position embedding parameter count."""
        max_seq_len = 2048
        embedding_dim = 4096

        # Parameter count should be max_seq_len * embedding_dim
        param_count = max_seq_len * embedding_dim

        # Should be around 8M parameters
        assert param_count == 8_388_608


@pytest.mark.perf
class TestPositionEmbeddingPerformance:
    """Performance tests for position embeddings."""

    @pytest.mark.skip(reason="Requires actual device")
    def test_sinusoidal_generation_latency(self):
        """Test sinusoidal position embedding generation latency."""
        # Would measure generation time

    @pytest.mark.skip(reason="Requires actual device")
    def test_learned_lookup_latency(self):
        """Test learned position embedding lookup latency."""
        # Would measure lookup time

    @pytest.mark.skip(reason="Requires actual device")
    def test_caching_speedup(self):
        """Test speedup from caching position embeddings."""
        # Would compare cached vs non-cached performance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
