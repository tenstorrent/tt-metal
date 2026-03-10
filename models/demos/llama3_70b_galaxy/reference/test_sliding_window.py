# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Tests for Sliding Window Attention reference implementation.

Run with: pytest models/demos/llama3_70b_galaxy/reference/test_sliding_window.py -v
"""

import pytest
import torch

from .sliding_window import (
    SlidingWindowConfig,
    create_causal_mask,
    create_sliding_window_mask,
    create_combined_mask,
    create_decode_mask,
    sliding_window_attention,
    compute_effective_context,
    get_attention_pattern_summary,
)


# ==============================================================================
# Fixtures
# ==============================================================================
@pytest.fixture
def olmo_config():
    """OLMo-3.1-32B sliding window configuration."""
    return SlidingWindowConfig()


@pytest.fixture
def small_config():
    """Small config for quick tests."""
    return SlidingWindowConfig(n_layers=8, sliding_window=4, pattern_size=4)


# ==============================================================================
# Layer Type Tests
# ==============================================================================
class TestLayerTypes:
    """Test layer type determination."""

    def test_layer_pattern_count(self, olmo_config):
        """Test correct count of layer types."""
        layer_types = olmo_config.get_all_layer_types()

        sliding_count = sum(1 for t in layer_types if t == "sliding_attention")
        full_count = sum(1 for t in layer_types if t == "full_attention")

        assert sliding_count == 48
        assert full_count == 16
        assert sliding_count + full_count == olmo_config.n_layers

    def test_layer_pattern_positions(self, olmo_config):
        """Test layer pattern: 3 sliding + 1 full."""
        for i in range(olmo_config.n_layers):
            layer_type = olmo_config.get_layer_type(i)
            expected = "full_attention" if (i + 1) % 4 == 0 else "sliding_attention"
            assert layer_type == expected, f"Layer {i} should be {expected}, got {layer_type}"

    def test_full_attention_layers(self, olmo_config):
        """Test positions of full attention layers."""
        full_layers = [i for i in range(olmo_config.n_layers) if olmo_config.get_layer_type(i) == "full_attention"]

        expected = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63]
        assert full_layers == expected

    def test_sliding_window_size(self, olmo_config):
        """Test sliding window size per layer."""
        for i in range(olmo_config.n_layers):
            window = olmo_config.get_sliding_window_size(i)
            if olmo_config.get_layer_type(i) == "full_attention":
                assert window is None
            else:
                assert window == olmo_config.sliding_window


# ==============================================================================
# Causal Mask Tests
# ==============================================================================
class TestCausalMask:
    """Test causal mask creation."""

    def test_shape(self):
        """Test mask shape."""
        mask = create_causal_mask(10)
        assert mask.shape == (10, 10)

    def test_diagonal_zero(self):
        """Test diagonal is 0 (can attend to self)."""
        mask = create_causal_mask(10)
        assert torch.all(torch.diag(mask) == 0)

    def test_upper_triangle_inf(self):
        """Test upper triangle is -inf (cannot attend to future)."""
        mask = create_causal_mask(10)
        # Check only strict upper triangle (above diagonal)
        for i in range(10):
            for j in range(i + 1, 10):
                assert mask[i, j] == float("-inf"), f"mask[{i},{j}] should be -inf"

    def test_lower_triangle_zero(self):
        """Test lower triangle is 0 (can attend to past)."""
        mask = create_causal_mask(10)
        lower = torch.tril(mask, diagonal=-1)
        assert torch.all(lower == 0)


# ==============================================================================
# Sliding Window Mask Tests
# ==============================================================================
class TestSlidingWindowMask:
    """Test sliding window mask creation."""

    def test_shape(self):
        """Test mask shape."""
        mask = create_sliding_window_mask(10, sliding_window=3)
        assert mask.shape == (10, 10)

    def test_includes_causal(self):
        """Test mask includes causal constraint."""
        mask = create_sliding_window_mask(10, sliding_window=3)

        # Future positions should be masked
        for i in range(10):
            for j in range(i + 1, 10):
                assert mask[i, j] == float("-inf")

    def test_window_constraint(self):
        """Test sliding window constraint."""
        seq_len = 10
        window = 3  # Window of 3 means attend to 3 tokens total
        mask = create_sliding_window_mask(seq_len, sliding_window=window)

        for i in range(seq_len):
            for j in range(seq_len):
                distance = i - j
                if j > i:
                    # Future: always masked
                    assert mask[i, j] == float("-inf"), f"mask[{i},{j}] should be -inf (future)"
                elif distance >= window:
                    # Outside window: masked (distance >= window means > window-1 back)
                    assert mask[i, j] == float("-inf"), f"mask[{i},{j}] should be -inf (outside window)"
                else:
                    # Within window and causal: not masked
                    assert mask[i, j] == 0, f"mask[{i},{j}] should be 0 (within window)"

    def test_first_positions_unrestricted(self):
        """Test first positions don't lose context (window > position)."""
        mask = create_sliding_window_mask(10, sliding_window=4)

        # Position 0 can only attend to position 0
        assert mask[0, 0] == 0

        # Position 3 can attend to 0, 1, 2, 3 (all within window)
        for j in range(4):
            assert mask[3, j] == 0

    def test_later_positions_restricted(self):
        """Test later positions are restricted to window."""
        mask = create_sliding_window_mask(10, sliding_window=3)

        # Position 8 can attend to 6, 7, 8 (window of 3 tokens)
        for j in range(6, 9):
            assert mask[8, j] == 0, f"mask[8,{j}] should be 0"

        # Position 8 cannot attend to 0-5
        for j in range(6):
            assert mask[8, j] == float("-inf"), f"mask[8,{j}] should be -inf"

    def test_effective_context(self):
        """Test effective context at each position."""
        window = 4
        mask = create_sliding_window_mask(10, sliding_window=window)

        for i in range(10):
            # Count attended positions
            attended = (mask[i] == 0).sum().item()
            expected = compute_effective_context(i, window)
            assert attended == expected, f"Position {i}: expected {expected}, got {attended}"


# ==============================================================================
# Combined Mask Tests
# ==============================================================================
class TestCombinedMask:
    """Test combined mask creation."""

    def test_full_attention_mask(self):
        """Test full attention mask (no sliding window)."""
        mask = create_combined_mask(8, sliding_window=None)

        assert mask.shape == (1, 1, 8, 8)

        # Should be regular causal mask
        causal = create_causal_mask(8)
        assert torch.all(mask[0, 0] == causal)

    def test_sliding_attention_mask(self):
        """Test sliding attention mask."""
        mask = create_combined_mask(8, sliding_window=3)

        assert mask.shape == (1, 1, 8, 8)

        # Should be sliding window mask
        sliding = create_sliding_window_mask(8, sliding_window=3)
        assert torch.all(mask[0, 0] == sliding)


# ==============================================================================
# Decode Mask Tests
# ==============================================================================
class TestDecodeMask:
    """Test decode-mode mask creation."""

    def test_shape(self):
        """Test decode mask shape."""
        mask = create_decode_mask(current_pos=5, kv_seq_len=6, sliding_window=3)
        assert mask.shape == (1, 1, 1, 6)

    def test_full_attention_decode(self):
        """Test decode without sliding window (full attention)."""
        mask = create_decode_mask(current_pos=7, kv_seq_len=8, sliding_window=None)

        # Should attend to all positions
        assert torch.all(mask == 0)

    def test_sliding_window_decode(self):
        """Test decode with sliding window."""
        window = 3
        current_pos = 7
        mask = create_decode_mask(current_pos=current_pos, kv_seq_len=8, sliding_window=window)

        # Should attend to positions 5, 6, 7 (window of 3 tokens)
        for j in range(8):
            distance = current_pos - j
            if distance >= window:
                assert mask[0, 0, 0, j] == float("-inf")
            else:
                assert mask[0, 0, 0, j] == 0

    def test_early_position_decode(self):
        """Test decode at early position (within window)."""
        mask = create_decode_mask(current_pos=2, kv_seq_len=3, sliding_window=4)

        # Window is larger than history, should attend to all
        assert torch.all(mask == 0)


# ==============================================================================
# Attention Tests
# ==============================================================================
class TestSlidingWindowAttention:
    """Test sliding window attention computation."""

    def test_output_shape_prefill(self):
        """Test output shape for prefill mode."""
        batch, n_heads, seq_len, head_dim = 2, 4, 32, 64

        q = torch.randn(batch, n_heads, seq_len, head_dim)
        k = torch.randn(batch, n_heads, seq_len, head_dim)
        v = torch.randn(batch, n_heads, seq_len, head_dim)

        output = sliding_window_attention(q, k, v, sliding_window=8)

        assert output.shape == (batch, n_heads, seq_len, head_dim)

    def test_output_shape_decode(self):
        """Test output shape for decode mode."""
        batch, n_heads, head_dim = 2, 4, 64
        kv_len = 32

        q = torch.randn(batch, n_heads, 1, head_dim)  # Single query
        k = torch.randn(batch, n_heads, kv_len, head_dim)
        v = torch.randn(batch, n_heads, kv_len, head_dim)

        output = sliding_window_attention(q, k, v, sliding_window=8)

        assert output.shape == (batch, n_heads, 1, head_dim)

    def test_full_vs_sliding(self):
        """Test that sliding window changes output."""
        torch.manual_seed(42)
        batch, n_heads, seq_len, head_dim = 1, 2, 16, 32

        q = torch.randn(batch, n_heads, seq_len, head_dim)
        k = torch.randn(batch, n_heads, seq_len, head_dim)
        v = torch.randn(batch, n_heads, seq_len, head_dim)

        out_full = sliding_window_attention(q, k, v, sliding_window=None)
        out_window = sliding_window_attention(q, k, v, sliding_window=4)

        # Outputs should be different due to masking
        assert not torch.allclose(out_full, out_window)

    def test_early_positions_match(self):
        """Test that early positions (within window) match full attention."""
        torch.manual_seed(42)
        batch, n_heads, seq_len, head_dim = 1, 2, 16, 32

        q = torch.randn(batch, n_heads, seq_len, head_dim)
        k = torch.randn(batch, n_heads, seq_len, head_dim)
        v = torch.randn(batch, n_heads, seq_len, head_dim)

        window = 8
        out_full = sliding_window_attention(q, k, v, sliding_window=None)
        out_window = sliding_window_attention(q, k, v, sliding_window=window)

        # First `window` positions should match (they see full context anyway)
        # Note: This is approximate due to softmax normalization effects
        early_full = out_full[:, :, :window]
        early_window = out_window[:, :, :window]
        assert torch.allclose(early_full, early_window, atol=1e-5)


# ==============================================================================
# Effective Context Tests
# ==============================================================================
class TestEffectiveContext:
    """Test effective context computation."""

    def test_early_positions(self):
        """Test context at positions less than window."""
        window = 4096
        assert compute_effective_context(0, window) == 1
        assert compute_effective_context(1, window) == 2
        assert compute_effective_context(100, window) == 101

    def test_window_boundary(self):
        """Test context at window boundary."""
        window = 4096
        assert compute_effective_context(4095, window) == 4096  # Exactly window
        assert compute_effective_context(4096, window) == 4096  # At window

    def test_beyond_window(self):
        """Test context beyond window."""
        window = 4096
        assert compute_effective_context(5000, window) == 4096
        assert compute_effective_context(10000, window) == 4096
        assert compute_effective_context(65535, window) == 4096  # Max OLMo position


# ==============================================================================
# Summary Tests
# ==============================================================================
class TestSummary:
    """Test summary generation."""

    def test_summary_keys(self, olmo_config):
        """Test summary has all keys."""
        summary = get_attention_pattern_summary(olmo_config)

        assert "total_layers" in summary
        assert "sliding_count" in summary
        assert "full_count" in summary
        assert "sliding_window" in summary
        assert "pattern" in summary

    def test_summary_values(self, olmo_config):
        """Test summary values are correct."""
        summary = get_attention_pattern_summary(olmo_config)

        assert summary["total_layers"] == 64
        assert summary["sliding_count"] == 48
        assert summary["full_count"] == 16
        assert summary["sliding_window"] == 4096


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
