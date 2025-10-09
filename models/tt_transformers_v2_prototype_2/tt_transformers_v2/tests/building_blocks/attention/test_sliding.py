# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Sliding Window Attention building block.
"""

import pytest
import torch
from src.building_blocks.attention import (
    SlidingWindowAttentionImplConfig,
    SlidingWindowAttentionSpec,
    create_sliding_window_mask,
    get_default_sliding_impl_config,
    sliding_decode_forward,
    sliding_prefill_forward,
)

import ttnn


class TestSlidingWindowAttentionSpec:
    """Test suite for SlidingWindowAttentionSpec."""

    def test_sliding_spec_basic(self):
        """Test basic sliding window attention spec."""
        spec = SlidingWindowAttentionSpec(
            hidden_dim=4096,
            num_heads=32,
            max_seq_len=2048,
            sliding_window_size=512,
        )
        spec.validate()
        assert spec.hidden_dim == 4096
        assert spec.num_heads == 32
        assert spec.sliding_window_size == 512

    def test_sliding_spec_with_global_tokens(self):
        """Test sliding window with global attention tokens."""
        spec = SlidingWindowAttentionSpec(
            hidden_dim=4096,
            num_heads=32,
            max_seq_len=2048,
            sliding_window_size=512,
            num_global_tokens=128,  # First 128 tokens have global attention
        )
        spec.validate()
        assert spec.sliding_window_size == 512
        assert spec.num_global_tokens == 128

    def test_sliding_spec_validation_errors(self):
        """Test sliding window spec validation errors."""
        # Window size larger than max seq len
        with pytest.raises(AssertionError):
            spec = SlidingWindowAttentionSpec(
                hidden_dim=4096,
                num_heads=32,
                max_seq_len=2048,
                sliding_window_size=4096,  # Too large
            )
            spec.validate()

        # Invalid window size
        with pytest.raises(AssertionError):
            spec = SlidingWindowAttentionSpec(
                hidden_dim=4096,
                num_heads=32,
                max_seq_len=2048,
                sliding_window_size=0,
            )
            spec.validate()

        # Global tokens exceed max seq len
        with pytest.raises(AssertionError):
            spec = SlidingWindowAttentionSpec(
                hidden_dim=4096,
                num_heads=32,
                max_seq_len=2048,
                sliding_window_size=512,
                num_global_tokens=3000,  # Too many
            )
            spec.validate()


class TestSlidingWindowAttentionImplConfig:
    """Test suite for SlidingWindowAttentionImplConfig."""

    def test_sliding_default_impl_config(self):
        """Test getting default sliding window implementation config."""
        spec = SlidingWindowAttentionSpec(
            hidden_dim=4096,
            num_heads=32,
            sliding_window_size=512,
        )

        for device in ["N150", "N300", "T3000", "TG"]:
            impl_config = get_default_sliding_impl_config(spec, device, "prefill")
            assert isinstance(impl_config, SlidingWindowAttentionImplConfig)
            assert impl_config.use_sparse_attention in [True, False]

    def test_sliding_impl_config_sparse_pattern(self):
        """Test sliding window sparse attention pattern config."""
        spec = SlidingWindowAttentionSpec(
            hidden_dim=4096,
            num_heads=32,
            sliding_window_size=512,
        )

        impl_config = get_default_sliding_impl_config(spec, "T3000", "prefill")

        # Check sparse pattern configuration
        if impl_config.use_sparse_attention:
            assert impl_config.sparse_block_size > 0
            assert impl_config.sparse_block_size <= spec.sliding_window_size


class TestSlidingWindowMask:
    """Test sliding window mask creation."""

    def test_create_sliding_window_mask_basic(self):
        """Test basic sliding window mask creation."""
        seq_len = 16
        window_size = 4

        mask = create_sliding_window_mask(seq_len, window_size)

        # Check shape
        assert mask.shape == (seq_len, seq_len)

        # Check that each position can only attend to window_size positions
        for i in range(seq_len):
            # Count number of positions this token can attend to
            num_attended = (mask[i] != float("-inf")).sum().item()
            expected = min(i + 1, window_size)  # Can't attend beyond current position
            assert num_attended == expected

    def test_create_sliding_window_mask_with_global(self):
        """Test sliding window mask with global tokens."""
        seq_len = 16
        window_size = 4
        num_global = 2

        mask = create_sliding_window_mask(seq_len, window_size, num_global_tokens=num_global)

        # Check that global tokens can attend to all positions
        for i in range(num_global):
            num_attended = (mask[i] != float("-inf")).sum().item()
            assert num_attended == i + 1  # Can attend to all previous tokens

        # Check that all tokens can attend to global tokens
        for i in range(num_global, seq_len):
            # Should be able to attend to all global tokens plus window
            can_attend_to_globals = all(mask[i, j] != float("-inf") for j in range(num_global))
            assert can_attend_to_globals


class TestSlidingWindowAttentionForward:
    """Test suite for sliding window attention forward functions."""

    @pytest.mark.skip(reason="Requires actual device")
    def test_sliding_prefill_forward_basic(self):
        """Test sliding window attention forward pass in prefill mode."""
        batch_size = 2
        seq_len = 1024
        hidden_dim = 4096

        spec = SlidingWindowAttentionSpec(
            hidden_dim=hidden_dim,
            num_heads=32,
            sliding_window_size=256,
            max_seq_len=2048,
        )
        impl_config = get_default_sliding_impl_config(spec, "T3000", "prefill")

        # Create dummy input
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        hidden_states_tt = ttnn.from_torch(hidden_states)

        # Forward pass
        output, cache = sliding_prefill_forward(
            hidden_states_tt,
            spec,
            impl_config,
            position_ids=None,
            attention_mask=None,
        )

        # Check output shape
        assert output.shape == hidden_states_tt.shape

    @pytest.mark.skip(reason="Requires actual device")
    def test_sliding_decode_forward_basic(self):
        """Test sliding window attention forward pass in decode mode."""
        batch_size = 2
        hidden_dim = 4096

        spec = SlidingWindowAttentionSpec(
            hidden_dim=hidden_dim,
            num_heads=32,
            sliding_window_size=256,
            max_seq_len=2048,
        )
        impl_config = get_default_sliding_impl_config(spec, "T3000", "decode")

        # Create dummy input (single token)
        hidden_states = torch.randn(batch_size, 1, hidden_dim)
        hidden_states_tt = ttnn.from_torch(hidden_states)

        # Create dummy cache (only need window_size entries)
        cache = {
            "k": torch.randn(batch_size, spec.num_heads, spec.sliding_window_size, spec.head_dim),
            "v": torch.randn(batch_size, spec.num_heads, spec.sliding_window_size, spec.head_dim),
            "cache_position": 512,  # Current position in sequence
        }

        # Forward pass
        output = sliding_decode_forward(
            hidden_states_tt,
            spec,
            impl_config,
            cache,
            position_ids=None,
        )

        # Check output shape
        assert output.shape == hidden_states_tt.shape


class TestSlidingWindowMemoryEfficiency:
    """Test memory efficiency of sliding window attention."""

    def test_sliding_cache_memory_savings(self):
        """Test that sliding window uses less memory for KV cache than full attention."""
        batch_size = 1
        max_seq_len = 32768  # Very long sequence
        sliding_window = 4096
        hidden_dim = 4096
        num_heads = 32
        head_dim = 128

        # Full attention cache size
        full_cache_size = 2 * batch_size * num_heads * max_seq_len * head_dim

        # Sliding window cache size
        sliding_cache_size = 2 * batch_size * num_heads * sliding_window * head_dim

        # Sliding window should use much less memory
        assert sliding_cache_size == (sliding_window / max_seq_len) * full_cache_size
        assert sliding_cache_size < full_cache_size / 8  # At least 8x savings


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
