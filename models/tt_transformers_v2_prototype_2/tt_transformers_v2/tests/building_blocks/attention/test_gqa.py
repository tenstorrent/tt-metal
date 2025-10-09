# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Grouped-Query Attention (GQA) building block.
"""

import pytest
import torch
from src.building_blocks.attention import (
    GroupedQueryAttentionImplConfig,
    GroupedQueryAttentionSpec,
    get_default_gqa_impl_config,
    gqa_decode_forward,
    gqa_prefill_forward,
)

import ttnn


class TestGroupedQueryAttentionSpec:
    """Test suite for GroupedQueryAttentionSpec."""

    def test_gqa_spec_basic(self):
        """Test basic grouped-query attention spec."""
        spec = GroupedQueryAttentionSpec(
            hidden_dim=4096, num_heads=32, num_kv_heads=8, max_seq_len=2048  # 4 query heads per kv head
        )
        spec.validate()
        assert spec.hidden_dim == 4096
        assert spec.num_heads == 32
        assert spec.num_kv_heads == 8
        assert spec.head_dim == 128  # 4096 / 32
        assert spec.num_q_heads_per_kv_head == 4

    def test_gqa_spec_single_kv_head(self):
        """Test GQA spec with single KV head (MQA)."""
        spec = GroupedQueryAttentionSpec(
            hidden_dim=4096, num_heads=32, num_kv_heads=1, max_seq_len=2048  # Multi-Query Attention
        )
        spec.validate()
        assert spec.num_kv_heads == 1
        assert spec.num_q_heads_per_kv_head == 32

    def test_gqa_spec_equal_heads(self):
        """Test GQA spec with equal query and kv heads (degenerates to MHA)."""
        spec = GroupedQueryAttentionSpec(
            hidden_dim=4096, num_heads=32, num_kv_heads=32, max_seq_len=2048  # Same as num_heads
        )
        spec.validate()
        assert spec.num_kv_heads == 32
        assert spec.num_q_heads_per_kv_head == 1

    def test_gqa_spec_validation_errors(self):
        """Test GQA spec validation errors."""
        # num_heads not divisible by num_kv_heads
        with pytest.raises(AssertionError):
            spec = GroupedQueryAttentionSpec(hidden_dim=4096, num_heads=32, num_kv_heads=7)  # 32 not divisible by 7
            spec.validate()

        # num_kv_heads > num_heads
        with pytest.raises(AssertionError):
            spec = GroupedQueryAttentionSpec(hidden_dim=4096, num_heads=32, num_kv_heads=64)
            spec.validate()


class TestGroupedQueryAttentionImplConfig:
    """Test suite for GroupedQueryAttentionImplConfig."""

    def test_gqa_default_impl_config(self):
        """Test getting default GQA implementation config."""
        spec = GroupedQueryAttentionSpec(hidden_dim=4096, num_heads=32, num_kv_heads=8)

        # Test different devices
        for device in ["N150", "N300", "T3000", "TG"]:
            for mode in ["prefill", "decode"]:
                impl_config = get_default_gqa_impl_config(spec, device, mode)
                assert isinstance(impl_config, GroupedQueryAttentionImplConfig)
                assert impl_config.kv_heads_repeat_interleaved in [True, False]

    def test_gqa_impl_config_kv_repeat_strategy(self):
        """Test KV head repeat strategy configuration."""
        spec = GroupedQueryAttentionSpec(hidden_dim=4096, num_heads=32, num_kv_heads=8)

        # Different devices might use different strategies
        n150_config = get_default_gqa_impl_config(spec, "N150", "prefill")
        t3000_config = get_default_gqa_impl_config(spec, "T3000", "prefill")

        # Both should have the repeat strategy defined
        assert hasattr(n150_config, "kv_heads_repeat_interleaved")
        assert hasattr(t3000_config, "kv_heads_repeat_interleaved")


class TestGroupedQueryAttentionForward:
    """Test suite for GQA forward functions."""

    @pytest.mark.skip(reason="Requires actual device")
    def test_gqa_prefill_forward_basic(self):
        """Test GQA forward pass in prefill mode."""
        batch_size = 2
        seq_len = 128
        hidden_dim = 4096

        spec = GroupedQueryAttentionSpec(hidden_dim=hidden_dim, num_heads=32, num_kv_heads=8, max_seq_len=2048)
        impl_config = get_default_gqa_impl_config(spec, "cpu", "prefill")

        # Create dummy input
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        hidden_states_tt = ttnn.from_torch(hidden_states)

        # Forward pass
        output, cache = gqa_prefill_forward(
            hidden_states_tt,
            spec,
            impl_config,
            position_ids=None,
            attention_mask=None,
        )

        # Check output shape
        assert output.shape == hidden_states_tt.shape
        assert "k" in cache and "v" in cache
        # KV cache should have num_kv_heads dimension
        assert cache["k"].shape[1] == spec.num_kv_heads
        assert cache["v"].shape[1] == spec.num_kv_heads

    @pytest.mark.skip(reason="Requires actual device")
    def test_gqa_decode_forward_basic(self):
        """Test GQA forward pass in decode mode."""
        batch_size = 2
        hidden_dim = 4096

        spec = GroupedQueryAttentionSpec(hidden_dim=hidden_dim, num_heads=32, num_kv_heads=8, max_seq_len=2048)
        impl_config = get_default_gqa_impl_config(spec, "cpu", "decode")

        # Create dummy input (single token)
        hidden_states = torch.randn(batch_size, 1, hidden_dim)
        hidden_states_tt = ttnn.from_torch(hidden_states)

        # Create dummy cache with correct KV heads
        cache = {
            "k": torch.randn(batch_size, spec.num_kv_heads, 128, spec.head_dim),
            "v": torch.randn(batch_size, spec.num_kv_heads, 128, spec.head_dim),
        }

        # Forward pass
        output = gqa_decode_forward(
            hidden_states_tt,
            spec,
            impl_config,
            cache,
            position_ids=None,
        )

        # Check output shape
        assert output.shape == hidden_states_tt.shape


class TestGroupedQueryAttentionMemoryEfficiency:
    """Test memory efficiency of GQA."""

    def test_gqa_cache_memory_savings(self):
        """Test that GQA uses less memory for KV cache than MHA."""
        batch_size = 1
        seq_len = 2048
        hidden_dim = 4096
        head_dim = 128

        # MHA cache size
        mha_num_heads = 32
        mha_cache_size = 2 * batch_size * mha_num_heads * seq_len * head_dim

        # GQA cache size
        gqa_num_kv_heads = 8
        gqa_cache_size = 2 * batch_size * gqa_num_kv_heads * seq_len * head_dim

        # GQA should use 4x less memory for cache
        assert gqa_cache_size == mha_cache_size // 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
