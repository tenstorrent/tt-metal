# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Multi-Head Attention (MHA) building block.
"""

import pytest
import torch
from src.building_blocks.attention import (
    MultiHeadAttentionImplConfig,
    MultiHeadAttentionSpec,
    get_default_mha_impl_config,
    mha_decode_forward,
    mha_prefill_forward,
)

import ttnn


class TestMultiHeadAttentionSpec:
    """Test suite for MultiHeadAttentionSpec."""

    def test_mha_spec_basic(self):
        """Test basic multi-head attention spec."""
        spec = MultiHeadAttentionSpec(hidden_dim=4096, num_heads=32, max_seq_len=2048)
        spec.validate()
        assert spec.hidden_dim == 4096
        assert spec.num_heads == 32
        assert spec.num_kv_heads == 32  # MHA: same as num_heads
        assert spec.head_dim == 128  # 4096 / 32
        assert spec.num_q_heads_per_kv_head == 1

    def test_mha_spec_custom_head_dim(self):
        """Test MHA spec with custom head dimension."""
        spec = MultiHeadAttentionSpec(hidden_dim=4096, num_heads=32, head_dim=128, max_seq_len=2048)
        spec.validate()
        assert spec.head_dim == 128

    def test_mha_spec_rope_params(self):
        """Test MHA spec with RoPE parameters."""
        spec = MultiHeadAttentionSpec(
            hidden_dim=4096, num_heads=32, max_seq_len=2048, rope_theta=10000.0, rope_scaling_factor=1.0
        )
        spec.validate()
        assert spec.rope_theta == 10000.0
        assert spec.rope_scaling_factor == 1.0

    def test_mha_spec_validation_errors(self):
        """Test MHA spec validation errors."""
        # Hidden dim not divisible by num_heads
        with pytest.raises(AssertionError):
            spec = MultiHeadAttentionSpec(hidden_dim=4095, num_heads=32)
            spec.validate()

        # Invalid hidden_dim
        with pytest.raises(AssertionError):
            spec = MultiHeadAttentionSpec(hidden_dim=0, num_heads=32)
            spec.validate()

        # Invalid num_heads
        with pytest.raises(AssertionError):
            spec = MultiHeadAttentionSpec(hidden_dim=4096, num_heads=0)
            spec.validate()


class TestMultiHeadAttentionImplConfig:
    """Test suite for MultiHeadAttentionImplConfig."""

    def test_mha_default_impl_config(self):
        """Test getting default MHA implementation config."""
        spec = MultiHeadAttentionSpec(hidden_dim=4096, num_heads=32)

        # Test different devices and modes
        for device in ["N150", "N300", "T3000", "TG"]:
            for mode in ["prefill", "decode"]:
                impl_config = get_default_mha_impl_config(spec, device, mode)
                assert isinstance(impl_config, MultiHeadAttentionImplConfig)
                assert impl_config.qkv_dtype in [ttnn.bfloat16, ttnn.bfloat8_b]
                assert impl_config.output_dtype in [ttnn.bfloat16, ttnn.bfloat8_b]

    def test_mha_impl_config_device_specific(self):
        """Test device-specific MHA configurations."""
        spec = MultiHeadAttentionSpec(hidden_dim=4096, num_heads=32)

        # N150 config
        n150_config = get_default_mha_impl_config(spec, "N150", "prefill")
        assert n150_config.use_flash_attention is True

        # T3000 config
        t3000_config = get_default_mha_impl_config(spec, "T3000", "prefill")
        assert t3000_config.compute_kernel_config is not None

    def test_mha_impl_config_mode_differences(self):
        """Test differences between prefill and decode configs."""
        spec = MultiHeadAttentionSpec(hidden_dim=4096, num_heads=32)

        prefill_config = get_default_mha_impl_config(spec, "N150", "prefill")
        decode_config = get_default_mha_impl_config(spec, "N150", "decode")

        # Cache dtype should be consistent
        assert prefill_config.cache_dtype == decode_config.cache_dtype
        # But flash attention might differ
        # (actual differences depend on implementation)


class TestMultiHeadAttentionForward:
    """Test suite for MHA forward functions."""

    @pytest.mark.skip(reason="Requires actual device")
    def test_mha_prefill_forward_basic(self):
        """Test MHA forward pass in prefill mode."""
        batch_size = 2
        seq_len = 128
        hidden_dim = 4096

        spec = MultiHeadAttentionSpec(hidden_dim=hidden_dim, num_heads=32, max_seq_len=2048)
        impl_config = get_default_mha_impl_config(spec, "cpu", "prefill")

        # Create dummy input
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        hidden_states_tt = ttnn.from_torch(hidden_states)

        # Forward pass
        output, cache = mha_prefill_forward(
            hidden_states_tt,
            spec,
            impl_config,
            position_ids=None,
            attention_mask=None,
        )

        # Check output shape
        assert output.shape == hidden_states_tt.shape
        assert "k" in cache and "v" in cache

    @pytest.mark.skip(reason="Requires actual device")
    def test_mha_decode_forward_basic(self):
        """Test MHA forward pass in decode mode."""
        batch_size = 2
        hidden_dim = 4096

        spec = MultiHeadAttentionSpec(hidden_dim=hidden_dim, num_heads=32, max_seq_len=2048)
        impl_config = get_default_mha_impl_config(spec, "cpu", "decode")

        # Create dummy input (single token)
        hidden_states = torch.randn(batch_size, 1, hidden_dim)
        hidden_states_tt = ttnn.from_torch(hidden_states)

        # Create dummy cache
        cache = {
            "k": torch.randn(batch_size, spec.num_heads, 128, spec.head_dim),
            "v": torch.randn(batch_size, spec.num_heads, 128, spec.head_dim),
        }

        # Forward pass
        output = mha_decode_forward(
            hidden_states_tt,
            spec,
            impl_config,
            cache,
            position_ids=None,
        )

        # Check output shape
        assert output.shape == hidden_states_tt.shape


@pytest.mark.perf
class TestMultiHeadAttentionPerformance:
    """Performance tests for MHA layers."""

    @pytest.mark.skip(reason="Requires actual device")
    @pytest.mark.parametrize("batch_size,seq_len", [(1, 128), (2, 512), (4, 2048)])
    def test_mha_prefill_latency(self, batch_size, seq_len):
        """Test MHA prefill latency meets targets."""
        # Implementation would measure actual latency

    @pytest.mark.skip(reason="Requires actual device")
    @pytest.mark.parametrize("batch_size", [1, 8, 32])
    def test_mha_decode_latency(self, batch_size):
        """Test MHA decode latency meets targets."""
        # Implementation would measure actual latency


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
