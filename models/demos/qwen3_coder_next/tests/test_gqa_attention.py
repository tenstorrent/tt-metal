# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for GQA attention (every 4th layer)."""

import pytest
import torch

from models.demos.qwen3_coder_next.tt.gqa_attention import GQAAttention
from models.demos.qwen3_coder_next.tt.model_config import Qwen3CoderNextConfig
from models.demos.qwen3_coder_next.tt.rope import precompute_freqs


class TestGQAAttention:
    @pytest.fixture
    def config(self):
        return Qwen3CoderNextConfig()

    @pytest.fixture
    def module(self, config):
        return GQAAttention(config, layer_idx=3)  # Layer 3 is a GQA layer

    def test_output_shape(self, config, module):
        """Output has correct shape."""
        x = torch.randn(2, 16, config.hidden_size)
        mask = GQAAttention.make_causal_mask(16, dtype=x.dtype)
        output, kv_cache = module(x, attention_mask=mask)
        assert output.shape == x.shape

    def test_kv_cache_shape(self, config, module):
        """KV cache has correct shape."""
        x = torch.randn(2, 16, config.hidden_size)
        _, (cached_k, cached_v) = module(x)
        assert cached_k.shape == (2, 16, config.num_key_value_heads, config.head_dim)
        assert cached_v.shape == (2, 16, config.num_key_value_heads, config.head_dim)

    def test_with_rope(self, config, module):
        """Works with partial RoPE."""
        x = torch.randn(2, 16, config.hidden_size)
        cos, sin = precompute_freqs(config.head_dim, 16, config.rope_theta, config.partial_rotary_factor)
        mask = GQAAttention.make_causal_mask(16, dtype=x.dtype)
        output, _ = module(x, cos=cos, sin=sin, attention_mask=mask)
        assert output.shape == x.shape

    def test_decode_with_kv_cache(self, config, module):
        """Decode mode: seq_len=1 with KV cache from prefill."""
        # Prefill
        x_prefill = torch.randn(2, 16, config.hidden_size)
        _, kv_cache = module(x_prefill)

        # Decode: single token
        x_decode = torch.randn(2, 1, config.hidden_size)
        output, new_kv = module(x_decode, kv_cache=kv_cache)
        assert output.shape == (2, 1, config.hidden_size)
        # KV cache should have grown by 1
        assert new_kv[0].shape[1] == 17

    def test_causal_mask(self):
        """Causal mask blocks future positions."""
        mask = GQAAttention.make_causal_mask(4)
        assert mask.shape == (1, 1, 4, 4)
        # Upper triangle should be -inf
        assert mask[0, 0, 0, 1] == float("-inf")
        assert mask[0, 0, 0, 0] == 0.0

    def test_deterministic(self, config, module):
        """Same input produces same output."""
        x = torch.randn(2, 16, config.hidden_size)
        mask = GQAAttention.make_causal_mask(16, dtype=x.dtype)
        out1, _ = module(x, attention_mask=mask)
        out2, _ = module(x, attention_mask=mask)
        torch.testing.assert_close(out1, out2)
