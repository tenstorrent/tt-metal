# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for MoE routing gate, expert MLP, and full MoE layer."""

import pytest
import torch

from models.demos.qwen3_coder_next.tt.expert_mlp import ExpertMLP, SharedExpert
from models.demos.qwen3_coder_next.tt.model_config import Qwen3CoderNextConfig
from models.demos.qwen3_coder_next.tt.moe import MoELayer
from models.demos.qwen3_coder_next.tt.moe_gate import MoEGate


class TestMoEGate:
    @pytest.fixture
    def config(self):
        return Qwen3CoderNextConfig()

    @pytest.fixture
    def gate(self, config):
        return MoEGate(config)

    def test_output_shapes(self, config, gate):
        """Gate outputs have correct shapes."""
        x = torch.randn(32, config.hidden_size)  # 32 tokens
        weights, indices = gate(x)
        assert weights.shape == (32, config.num_experts_per_tok)
        assert indices.shape == (32, config.num_experts_per_tok)

    def test_weights_normalized(self, config, gate):
        """Top-K weights sum to 1 when norm_topk_prob is True."""
        x = torch.randn(16, config.hidden_size)
        weights, _ = gate(x)
        sums = weights.float().sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-5, rtol=1e-5)

    def test_indices_valid(self, config, gate):
        """Expert indices are in valid range."""
        x = torch.randn(16, config.hidden_size)
        _, indices = gate(x)
        assert indices.min() >= 0
        assert indices.max() < config.num_experts

    def test_indices_unique_per_token(self, config, gate):
        """Each token routes to K unique experts."""
        x = torch.randn(16, config.hidden_size)
        _, indices = gate(x)
        for i in range(16):
            assert len(set(indices[i].tolist())) == config.num_experts_per_tok


class TestExpertMLP:
    def test_output_shape(self):
        """Expert MLP preserves hidden size."""
        mlp = ExpertMLP(hidden_size=2048, intermediate_size=512)
        x = torch.randn(8, 2048)
        out = mlp(x)
        assert out.shape == (8, 2048)

    def test_different_intermediate_sizes(self):
        """Works with various intermediate sizes."""
        for isize in [256, 512, 1024]:
            mlp = ExpertMLP(hidden_size=2048, intermediate_size=isize)
            x = torch.randn(4, 2048)
            out = mlp(x)
            assert out.shape == (4, 2048)


class TestSharedExpert:
    def test_output_shape(self):
        """Shared expert preserves hidden size."""
        config = Qwen3CoderNextConfig()
        expert = SharedExpert(config)
        x = torch.randn(8, config.hidden_size)
        out = expert(x)
        assert out.shape == (8, config.hidden_size)


class TestMoELayer:
    @pytest.fixture
    def config(self):
        # Use smaller config for testing to avoid OOM (512 experts is large)
        config = Qwen3CoderNextConfig.__new__(Qwen3CoderNextConfig)
        config.hidden_size = 64
        config.num_hidden_layers = 4
        config.num_attention_heads = 4
        config.num_key_value_heads = 2
        config.head_dim = 16
        config.vocab_size = 1000
        config.max_position_embeddings = 512
        config.intermediate_size = 128
        config.moe_intermediate_size = 32
        config.num_experts = 8  # Small for testing
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

    @pytest.fixture
    def moe(self, config):
        return MoELayer(config)

    def test_output_shape(self, config, moe):
        """MoE preserves input shape."""
        x = torch.randn(2, 4, config.hidden_size)
        out = moe(x)
        assert out.shape == x.shape

    def test_gradient_flows(self, config, moe):
        """Gradients flow through MoE layer."""
        x = torch.randn(2, 4, config.hidden_size, requires_grad=True)
        out = moe(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_deterministic(self, config, moe):
        """Same input produces same output."""
        x = torch.randn(2, 4, config.hidden_size)
        out1 = moe(x)
        out2 = moe(x)
        torch.testing.assert_close(out1, out2)
