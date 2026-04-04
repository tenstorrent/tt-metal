# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for hybrid decoder block."""

import pytest
import torch

from models.demos.qwen3_coder_next.tt.decoder_block import HybridDecoderBlock
from models.demos.qwen3_coder_next.tt.gqa_attention import GQAAttention
from models.demos.qwen3_coder_next.tt.model_config import Qwen3CoderNextConfig
from models.demos.qwen3_coder_next.tt.rope import precompute_freqs


def make_small_config():
    """Create a small config for fast testing."""
    config = Qwen3CoderNextConfig.__new__(Qwen3CoderNextConfig)
    config.hidden_size = 64
    config.num_hidden_layers = 8
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 16
    config.vocab_size = 1000
    config.max_position_embeddings = 512
    config.intermediate_size = 128
    config.moe_intermediate_size = 32
    config.num_experts = 8
    config.num_experts_per_tok = 2
    config.shared_expert_intermediate_size = 32
    config.hidden_act = "silu"
    config.router_aux_loss_coef = 0.001
    config.norm_topk_prob = True
    config.full_attention_interval = 4
    config.partial_rotary_factor = 0.25
    config.rope_theta = 5000000.0
    config.linear_key_head_dim = 8
    config.linear_num_key_heads = 4
    config.linear_num_value_heads = 4
    config.linear_conv_kernel_dim = 4
    config.rms_norm_eps = 1e-6
    config.model_type = "qwen3_next"
    config.architectures = ["Qwen3NextForCausalLM"]
    return config


class TestHybridDecoderBlock:
    @pytest.fixture
    def config(self):
        return make_small_config()

    def test_deltanet_layer(self, config):
        """Layer 0 should use DeltaNet attention."""
        block = HybridDecoderBlock(config, layer_idx=0)
        assert not block.is_gqa_layer
        x = torch.randn(2, 8, config.hidden_size)
        output, state = block(x)
        assert output.shape == x.shape
        assert "recurrent_state" in state

    def test_gqa_layer(self, config):
        """Layer 3 should use GQA attention."""
        block = HybridDecoderBlock(config, layer_idx=3)
        assert block.is_gqa_layer
        x = torch.randn(2, 8, config.hidden_size)
        cos, sin = precompute_freqs(config.head_dim, 8, config.rope_theta, config.partial_rotary_factor)
        mask = GQAAttention.make_causal_mask(8, dtype=x.dtype)
        output, state = block(x, cos=cos, sin=sin, attention_mask=mask)
        assert output.shape == x.shape
        assert "kv_cache" in state

    def test_layer_type_pattern(self, config):
        """Verify correct DeltaNet/GQA assignment pattern."""
        for i in range(8):
            block = HybridDecoderBlock(config, layer_idx=i)
            expected_gqa = i % 4 == 3
            assert block.is_gqa_layer == expected_gqa, f"Layer {i}: expected GQA={expected_gqa}"

    def test_residual_connection(self, config):
        """Output should be different from input (residual + transform)."""
        block = HybridDecoderBlock(config, layer_idx=0)
        x = torch.randn(2, 4, config.hidden_size)
        output, _ = block(x)
        assert not torch.allclose(output, x)

    def test_deterministic(self, config):
        """Same input gives same output."""
        block = HybridDecoderBlock(config, layer_idx=0)
        x = torch.randn(2, 4, config.hidden_size)
        out1, _ = block(x)
        out2, _ = block(x)
        torch.testing.assert_close(out1, out2)
