# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Token Embedding building block.
"""

import pytest
import torch
from src.building_blocks.embeddings import (
    TokenEmbeddingImplConfig,
    TokenEmbeddingSpec,
    get_default_token_impl_config,
    token_embedding_forward,
)

import ttnn


class TestTokenEmbeddingSpec:
    """Test suite for TokenEmbeddingSpec."""

    def test_token_embedding_spec_basic(self):
        """Test basic token embedding spec creation and validation."""
        spec = TokenEmbeddingSpec(vocab_size=32000, embedding_dim=4096)
        spec.validate()
        assert spec.vocab_size == 32000
        assert spec.embedding_dim == 4096
        assert spec.padding_idx is None
        assert spec.max_norm is None

    def test_token_embedding_spec_with_padding(self):
        """Test token embedding spec with padding index."""
        spec = TokenEmbeddingSpec(vocab_size=32000, embedding_dim=4096, padding_idx=0)
        spec.validate()
        assert spec.padding_idx == 0

    def test_token_embedding_spec_with_max_norm(self):
        """Test token embedding spec with max norm constraint."""
        spec = TokenEmbeddingSpec(vocab_size=32000, embedding_dim=4096, max_norm=1.0)
        spec.validate()
        assert spec.max_norm == 1.0

    def test_token_embedding_spec_large_vocab(self):
        """Test token embedding spec with large vocabulary."""
        spec = TokenEmbeddingSpec(vocab_size=128000, embedding_dim=8192)  # Large vocab like GPT
        spec.validate()
        assert spec.vocab_size == 128000

    def test_token_embedding_spec_validation_errors(self):
        """Test token embedding spec validation errors."""
        # Invalid vocab_size
        with pytest.raises(AssertionError):
            spec = TokenEmbeddingSpec(vocab_size=0, embedding_dim=4096)
            spec.validate()

        with pytest.raises(AssertionError):
            spec = TokenEmbeddingSpec(vocab_size=-1, embedding_dim=4096)
            spec.validate()

        # Invalid embedding_dim
        with pytest.raises(AssertionError):
            spec = TokenEmbeddingSpec(vocab_size=32000, embedding_dim=0)
            spec.validate()

        # Invalid padding_idx
        with pytest.raises(AssertionError):
            spec = TokenEmbeddingSpec(vocab_size=32000, embedding_dim=4096, padding_idx=32000)  # Out of range
            spec.validate()

        with pytest.raises(AssertionError):
            spec = TokenEmbeddingSpec(vocab_size=32000, embedding_dim=4096, padding_idx=-2)  # Negative (except -1)
            spec.validate()


class TestTokenEmbeddingImplConfig:
    """Test suite for TokenEmbeddingImplConfig."""

    def test_token_embedding_default_impl_config(self):
        """Test getting default implementation config for token embedding."""
        spec = TokenEmbeddingSpec(vocab_size=32000, embedding_dim=4096)

        # Test different devices
        for device in ["N150", "N300", "T3000", "TG"]:
            impl_config = get_default_token_impl_config(spec, device)
            assert isinstance(impl_config, TokenEmbeddingImplConfig)
            assert impl_config.dtype in [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.float32]

    def test_token_embedding_impl_config_on_device(self):
        """Test on-device embedding table configuration."""
        spec = TokenEmbeddingSpec(vocab_size=32000, embedding_dim=4096)

        # Hardware devices should use on-device embedding table
        for device in ["N150", "N300", "T3000"]:
            impl_config = get_default_token_impl_config(spec, device)
            assert impl_config.use_embedding_table_on_device is True

    def test_token_embedding_impl_config_memory(self):
        """Test token embedding memory configuration."""
        spec = TokenEmbeddingSpec(vocab_size=128000, embedding_dim=8192)  # Large

        impl_config = get_default_token_impl_config(spec, "TG")

        # Should have appropriate memory config for large embeddings
        assert impl_config.embedding_table_memory_config is not None


class TestTokenEmbeddingForward:
    """Test suite for token embedding forward function."""

    @pytest.mark.skip(reason="Requires actual device")
    def test_token_embedding_forward_basic(self):
        """Test token embedding forward pass with basic inputs."""
        batch_size = 2
        seq_len = 128
        vocab_size = 32000
        embedding_dim = 4096

        spec = TokenEmbeddingSpec(vocab_size=vocab_size, embedding_dim=embedding_dim)
        impl_config = get_default_token_impl_config(spec, "cpu")

        # Create dummy input tokens
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        input_ids_tt = ttnn.from_torch(input_ids)

        # Forward pass
        output = token_embedding_forward(input_ids_tt, spec, impl_config)

        # Check output shape
        assert output.shape == (batch_size, seq_len, embedding_dim)

    @pytest.mark.skip(reason="Requires actual device")
    def test_token_embedding_forward_with_padding(self):
        """Test token embedding forward pass with padding tokens."""
        batch_size = 2
        seq_len = 128
        vocab_size = 32000
        embedding_dim = 4096
        padding_idx = 0

        spec = TokenEmbeddingSpec(vocab_size=vocab_size, embedding_dim=embedding_dim, padding_idx=padding_idx)
        impl_config = get_default_token_impl_config(spec, "cpu")

        # Create input with padding tokens
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        input_ids[0, -10:] = padding_idx  # Last 10 tokens are padding
        input_ids_tt = ttnn.from_torch(input_ids)

        # Forward pass
        output = token_embedding_forward(input_ids_tt, spec, impl_config)

        # Padding embeddings should be zeros (if implemented correctly)
        output_torch = ttnn.to_torch(output)
        padding_embeddings = output_torch[0, -10:, :]
        assert torch.allclose(padding_embeddings, torch.zeros_like(padding_embeddings))

    @pytest.mark.skip(reason="Requires actual device")
    def test_token_embedding_forward_out_of_vocab(self):
        """Test token embedding behavior with out-of-vocabulary tokens."""
        batch_size = 1
        seq_len = 16
        vocab_size = 32000
        embedding_dim = 4096

        spec = TokenEmbeddingSpec(vocab_size=vocab_size, embedding_dim=embedding_dim)
        impl_config = get_default_token_impl_config(spec, "cpu")

        # Create input with out-of-vocab token
        input_ids = torch.full((batch_size, seq_len), vocab_size)  # OOV
        input_ids_tt = ttnn.from_torch(input_ids)

        # Should raise error or handle gracefully
        with pytest.raises(Exception):
            token_embedding_forward(input_ids_tt, spec, impl_config)


@pytest.mark.perf
class TestTokenEmbeddingPerformance:
    """Performance tests for token embeddings."""

    @pytest.mark.skip(reason="Requires actual device")
    @pytest.mark.parametrize(
        "vocab_size,embedding_dim",
        [
            (32000, 768),  # Small model
            (50000, 1024),  # Medium model
            (128000, 4096),  # Large model
        ],
    )
    def test_token_embedding_lookup_latency(self, vocab_size, embedding_dim):
        """Test token embedding lookup latency for various sizes."""
        # Would measure actual latency

    @pytest.mark.skip(reason="Requires actual device")
    def test_token_embedding_memory_usage(self):
        """Test token embedding table memory usage."""
        # Would measure memory consumption


class TestTokenEmbeddingScaling:
    """Test token embedding scaling properties."""

    def test_embedding_table_size_calculation(self):
        """Test embedding table size calculations."""
        vocab_size = 32000
        embedding_dim = 4096
        dtype_bytes = 2  # bfloat16

        # Calculate expected size
        expected_size_mb = (vocab_size * embedding_dim * dtype_bytes) / (1024 * 1024)

        # Should be around 250 MB for these parameters
        assert 240 < expected_size_mb < 260


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
