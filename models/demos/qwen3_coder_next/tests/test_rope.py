# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for partial RoPE implementation."""

import torch

from models.demos.qwen3_coder_next.reference.functional import reference_partial_rope
from models.demos.qwen3_coder_next.tt.model_config import Qwen3CoderNextConfig
from models.demos.qwen3_coder_next.tt.rope import PartialRoPE, apply_partial_rope_torch, precompute_freqs


class TestPrecomputeFreqs:
    def test_output_shapes(self):
        """Verify cos/sin have correct shapes."""
        cos, sin = precompute_freqs(head_dim=256, max_seq_len=1024, rope_theta=5000000.0, partial_rotary_factor=0.25)
        # rotary_dim = 256 * 0.25 = 64, so output is (seq_len, rotary_dim/2) = (1024, 32)
        assert cos.shape == (1024, 32)
        assert sin.shape == (1024, 32)

    def test_values_bounded(self):
        """Verify cos/sin values are in [-1, 1]."""
        cos, sin = precompute_freqs(head_dim=256, max_seq_len=512)
        assert cos.abs().max() <= 1.0
        assert sin.abs().max() <= 1.0

    def test_position_zero(self):
        """At position 0, cos should be 1 and sin should be 0."""
        cos, sin = precompute_freqs(head_dim=256, max_seq_len=16)
        torch.testing.assert_close(cos[0], torch.ones_like(cos[0]), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(sin[0], torch.zeros_like(sin[0]), atol=1e-6, rtol=1e-6)


class TestApplyPartialRope:
    def test_output_shape_preserved(self):
        """Output shape should match input shape."""
        x = torch.randn(2, 16, 16, 256)
        cos, sin = precompute_freqs(head_dim=256, max_seq_len=16)
        out = apply_partial_rope_torch(x, cos, sin, partial_rotary_factor=0.25)
        assert out.shape == x.shape

    def test_passthrough_dims_unchanged(self):
        """Non-rotary dimensions (75%) should be unchanged."""
        x = torch.randn(2, 16, 16, 256)
        cos, sin = precompute_freqs(head_dim=256, max_seq_len=16)
        out = apply_partial_rope_torch(x, cos, sin, partial_rotary_factor=0.25)
        # Last 192 dims (75%) should be identical
        torch.testing.assert_close(out[..., 64:], x[..., 64:])

    def test_rotary_dims_modified(self):
        """Rotary dimensions (25%) should be modified (except at position 0)."""
        x = torch.randn(2, 16, 16, 256)
        cos, sin = precompute_freqs(head_dim=256, max_seq_len=16)
        out = apply_partial_rope_torch(x, cos, sin, partial_rotary_factor=0.25)
        # At position > 0, first 64 dims should differ
        assert not torch.allclose(out[:, 1:, :, :64], x[:, 1:, :, :64])

    def test_matches_reference(self):
        """TT partial RoPE should match reference implementation."""
        x = torch.randn(2, 16, 16, 256)
        cos, sin = precompute_freqs(head_dim=256, max_seq_len=16)

        out_tt = apply_partial_rope_torch(x, cos, sin, partial_rotary_factor=0.25)
        out_ref = reference_partial_rope(x, cos, sin, partial_rotary_factor=0.25)

        torch.testing.assert_close(out_tt, out_ref, atol=1e-5, rtol=1e-5)


class TestPartialRoPEManager:
    def test_init(self):
        """PartialRoPE initializes correctly from config."""
        config = Qwen3CoderNextConfig()
        rope = PartialRoPE(config, max_seq_len=1024)
        assert rope.rotary_dim == 64
        assert rope.non_rotary_dim == 192
        assert rope.cos.shape == (1024, 32)

    def test_get_cos_sin(self):
        """get_cos_sin returns correct slices."""
        config = Qwen3CoderNextConfig()
        rope = PartialRoPE(config, max_seq_len=1024)
        cos, sin = rope.get_cos_sin(seq_len=128)
        assert cos.shape == (128, 32)
        assert sin.shape == (128, 32)

    def test_apply(self):
        """apply() rotates Q and K correctly."""
        config = Qwen3CoderNextConfig()
        rope = PartialRoPE(config, max_seq_len=128)

        q = torch.randn(2, 16, config.num_attention_heads, config.head_dim)
        k = torch.randn(2, 16, config.num_key_value_heads, config.head_dim)
        cos, sin = rope.get_cos_sin(seq_len=16)

        q_rot, k_rot = rope.apply(q, k, cos, sin)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
        # Pass-through dims unchanged
        torch.testing.assert_close(q_rot[..., 64:], q[..., 64:])
        torch.testing.assert_close(k_rot[..., 64:], k[..., 64:])
