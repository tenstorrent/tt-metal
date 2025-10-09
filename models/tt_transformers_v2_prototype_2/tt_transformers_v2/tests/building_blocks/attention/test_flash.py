# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for Flash Attention building block.
"""

import pytest
import torch
from src.building_blocks.attention import (
    FlashAttentionImplConfig,
    FlashAttentionSpec,
    flash_prefill_forward,
    get_default_flash_impl_config,
)

import ttnn


class TestFlashAttentionSpec:
    """Test suite for FlashAttentionSpec."""

    def test_flash_spec_basic(self):
        """Test basic flash attention spec."""
        spec = FlashAttentionSpec(
            hidden_dim=4096,
            num_heads=32,
            max_seq_len=2048,
            window_size=256,  # Flash attention block size
        )
        spec.validate()
        assert spec.hidden_dim == 4096
        assert spec.num_heads == 32
        assert spec.window_size == 256
        assert spec.use_causal_mask is True  # Default

    def test_flash_spec_with_gqa(self):
        """Test flash attention spec with GQA."""
        spec = FlashAttentionSpec(
            hidden_dim=4096,
            num_heads=32,
            num_kv_heads=8,
            max_seq_len=2048,
            window_size=256,
        )
        spec.validate()
        assert spec.num_kv_heads == 8
        assert spec.num_q_heads_per_kv_head == 4

    def test_flash_spec_non_causal(self):
        """Test flash attention spec for non-causal (encoder) attention."""
        spec = FlashAttentionSpec(
            hidden_dim=4096,
            num_heads=32,
            max_seq_len=2048,
            window_size=256,
            use_causal_mask=False,
        )
        spec.validate()
        assert spec.use_causal_mask is False

    def test_flash_spec_validation_errors(self):
        """Test flash attention spec validation errors."""
        # Window size larger than max seq len
        with pytest.raises(AssertionError):
            spec = FlashAttentionSpec(
                hidden_dim=4096,
                num_heads=32,
                max_seq_len=2048,
                window_size=4096,  # Too large
            )
            spec.validate()

        # Window size not power of 2 (if required)
        # Note: This might be implementation specific


class TestFlashAttentionImplConfig:
    """Test suite for FlashAttentionImplConfig."""

    def test_flash_default_impl_config(self):
        """Test getting default flash attention implementation config."""
        spec = FlashAttentionSpec(
            hidden_dim=4096,
            num_heads=32,
            window_size=256,
        )

        # Test different devices
        for device in ["N150", "N300", "T3000", "TG"]:
            impl_config = get_default_flash_impl_config(spec, device, "prefill")
            assert isinstance(impl_config, FlashAttentionImplConfig)
            assert impl_config.block_size > 0
            assert impl_config.num_blocks > 0

    def test_flash_impl_config_block_params(self):
        """Test flash attention block parameters."""
        spec = FlashAttentionSpec(
            hidden_dim=4096,
            num_heads=32,
            window_size=256,
        )

        impl_config = get_default_flash_impl_config(spec, "T3000", "prefill")

        # Check block configuration
        assert impl_config.block_size <= spec.window_size
        assert impl_config.softmax_algorithm in ["stable", "online"]
        assert impl_config.use_fused_qkv in [True, False]

    def test_flash_impl_config_memory_efficiency(self):
        """Test flash attention memory configuration."""
        spec = FlashAttentionSpec(
            hidden_dim=4096,
            num_heads=32,
            window_size=256,
            max_seq_len=8192,  # Long sequence
        )

        impl_config = get_default_flash_impl_config(spec, "T3000", "prefill")

        # Flash attention should enable processing long sequences
        assert impl_config.enable_tiling is True
        assert impl_config.tile_size > 0


class TestFlashAttentionForward:
    """Test suite for flash attention forward functions."""

    @pytest.mark.skip(reason="Requires actual device")
    def test_flash_prefill_forward_basic(self):
        """Test flash attention forward pass in prefill mode."""
        batch_size = 2
        seq_len = 512
        hidden_dim = 4096

        spec = FlashAttentionSpec(
            hidden_dim=hidden_dim,
            num_heads=32,
            window_size=256,
            max_seq_len=2048,
        )
        impl_config = get_default_flash_impl_config(spec, "T3000", "prefill")

        # Create dummy input
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        hidden_states_tt = ttnn.from_torch(hidden_states)

        # Forward pass
        output, cache = flash_prefill_forward(
            hidden_states_tt,
            spec,
            impl_config,
            position_ids=None,
            attention_mask=None,
        )

        # Check output shape
        assert output.shape == hidden_states_tt.shape

    @pytest.mark.skip(reason="Requires actual device")
    def test_flash_prefill_forward_long_sequence(self):
        """Test flash attention with long sequences."""
        batch_size = 1
        seq_len = 8192  # Long sequence
        hidden_dim = 4096

        spec = FlashAttentionSpec(
            hidden_dim=hidden_dim,
            num_heads=32,
            window_size=256,
            max_seq_len=16384,
        )
        impl_config = get_default_flash_impl_config(spec, "T3000", "prefill")

        # Create dummy input
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        hidden_states_tt = ttnn.from_torch(hidden_states)

        # Forward pass should handle long sequence efficiently
        output, cache = flash_prefill_forward(
            hidden_states_tt,
            spec,
            impl_config,
            position_ids=None,
            attention_mask=None,
        )

        assert output.shape == hidden_states_tt.shape


@pytest.mark.perf
class TestFlashAttentionPerformance:
    """Performance tests for flash attention."""

    @pytest.mark.skip(reason="Requires actual device")
    @pytest.mark.parametrize("seq_len", [512, 2048, 8192])
    def test_flash_vs_standard_attention_speedup(self, seq_len):
        """Test that flash attention provides speedup over standard attention."""
        # Would compare flash attention vs standard attention timings

    @pytest.mark.skip(reason="Requires actual device")
    def test_flash_memory_usage(self):
        """Test that flash attention uses less memory than standard attention."""
        # Would measure peak memory usage


class TestFlashAttentionNumerics:
    """Numerical accuracy tests for flash attention."""

    @pytest.mark.skip(reason="Requires actual device")
    def test_flash_attention_accuracy(self):
        """Test flash attention numerical accuracy against reference."""
        batch_size = 1
        seq_len = 256
        hidden_dim = 512
        num_heads = 8

        # Create reference implementation
        # Compare flash attention output with standard attention
        # Check that results are close within tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
