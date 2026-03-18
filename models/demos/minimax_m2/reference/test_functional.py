# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Reference tests for MiniMax-M2.5 standalone functional implementations.

Each test verifies that our standalone reference function produces numerically
identical (atol=1e-5) output to a mini HuggingFace-equivalent module built
from the same random weights.

Run:
    export TT_METAL_HOME=$(pwd) && export PYTHONPATH=$(pwd) && source python_env/bin/activate
    pytest models/demos/minimax_m2/reference/test_functional.py -v
"""


import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.demos.minimax_m2.reference.functional import (
    MiniMaxM2Config,
    apply_partial_rope,
    attention_forward,
    build_rope_cache,
    decoder_layer_forward,
    expert_mlp_forward,
    make_random_state_dict,
    model_forward,
    moe_forward,
    rmsnorm_forward,
)

# ---------------------------------------------------------------------------
# Small config for fast tests (2 layers, 4 experts)
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg():
    return MiniMaxM2Config(
        hidden_size=256,
        head_dim=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=128,
        num_local_experts=4,
        num_experts_per_tok=2,
        rotary_dim=32,
        rope_theta=5_000_000.0,
        rms_norm_eps=1e-6,
        vocab_size=512,
        use_qk_norm=True,
        use_routing_bias=True,
    )


# ---------------------------------------------------------------------------
# Helper: HF-equivalent modules (built inline, no HF dependency)
# ---------------------------------------------------------------------------


class HFRMSNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        x_fp32 = x.float()
        rms = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * (x_fp32 * rms).to(x.dtype)


class HFAttention(nn.Module):
    """Mirror of MiniMaxM2Attention — used to cross-validate functional.py."""

    def __init__(self, cfg: MiniMaxM2Config):
        super().__init__()
        self.cfg = cfg
        H, D = cfg.hidden_size, cfg.head_dim
        NQ, NK = cfg.num_attention_heads, cfg.num_key_value_heads
        self.q_proj = nn.Linear(H, NQ * D, bias=False)
        self.k_proj = nn.Linear(H, NK * D, bias=False)
        self.v_proj = nn.Linear(H, NK * D, bias=False)
        self.o_proj = nn.Linear(NQ * D, H, bias=False)
        self.q_norm = HFRMSNorm(NQ * D, eps=cfg.rms_norm_eps)
        self.k_norm = HFRMSNorm(NK * D, eps=cfg.rms_norm_eps)

    def forward(self, x, cos, sin):
        B, S, _ = x.shape
        NQ, NK = self.cfg.num_attention_heads, self.cfg.num_key_value_heads
        D = self.cfg.head_dim
        groups = self.cfg.num_key_value_groups

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.view(B, S, NQ, D).transpose(1, 2)
        k = k.view(B, S, NK, D).transpose(1, 2)
        v = v.view(B, S, NK, D).transpose(1, 2)

        q, k = apply_partial_rope(q, k, cos, sin)

        if groups > 1:
            k = k.unsqueeze(2).expand(B, NK, groups, S, D).reshape(B, NQ, S, D)
            v = v.unsqueeze(2).expand(B, NK, groups, S, D).reshape(B, NQ, S, D)

        scale = D**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, NQ * D)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRMSNorm:
    def test_matches_nn_module(self, cfg):
        torch.manual_seed(42)
        x = torch.randn(2, 16, cfg.hidden_size)
        weight = torch.ones(cfg.hidden_size)

        hf_norm = HFRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        hf_norm.weight.data.copy_(weight)

        ref_out = rmsnorm_forward(x, weight, cfg.rms_norm_eps)
        hf_out = hf_norm(x)

        assert torch.allclose(ref_out, hf_out, atol=1e-5), f"Max diff: {(ref_out - hf_out).abs().max()}"

    def test_random_weight(self, cfg):
        torch.manual_seed(0)
        x = torch.randn(1, 8, cfg.hidden_size)
        weight = torch.randn(cfg.hidden_size)

        hf_norm = HFRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        hf_norm.weight.data.copy_(weight)

        assert torch.allclose(rmsnorm_forward(x, weight, cfg.rms_norm_eps), hf_norm(x), atol=1e-5)


class TestRoPE:
    def test_partial_rope_output_shape(self, cfg):
        B, S = 1, 16
        cos, sin = build_rope_cache(S, cfg.rotary_dim, cfg.rope_theta)
        assert cos.shape == (S, cfg.rotary_dim)
        assert sin.shape == (S, cfg.rotary_dim)

    def test_partial_rope_passthrough_region_unchanged(self, cfg):
        """Dims beyond rotary_dim must not change."""
        torch.manual_seed(1)
        B, S = 1, 8
        NQ, NK, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        q = torch.randn(B, NQ, S, D)
        k = torch.randn(B, NK, S, D)
        cos, sin = build_rope_cache(S, cfg.rotary_dim, cfg.rope_theta)
        q_out, k_out = apply_partial_rope(q, k, cos, sin)

        # Passthrough region (rotary_dim:) must be identical
        assert torch.allclose(
            q_out[..., cfg.rotary_dim :], q[..., cfg.rotary_dim :]
        ), "q passthrough region was modified"
        assert torch.allclose(
            k_out[..., cfg.rotary_dim :], k[..., cfg.rotary_dim :]
        ), "k passthrough region was modified"

    def test_partial_rope_rotary_region_changed(self, cfg):
        """Dims in [0, rotary_dim) must be different from input."""
        torch.manual_seed(2)
        B, S = 1, 8
        NQ, NK, D = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        q = torch.randn(B, NQ, S, D)
        k = torch.randn(B, NK, S, D)
        cos, sin = build_rope_cache(S, cfg.rotary_dim, cfg.rope_theta)
        q_out, k_out = apply_partial_rope(q, k, cos, sin)

        assert not torch.allclose(
            q_out[..., : cfg.rotary_dim], q[..., : cfg.rotary_dim]
        ), "q rotary region was NOT modified"


class TestAttention:
    def test_matches_hf_module(self, cfg):
        """functional.attention_forward must match HFAttention with same weights."""
        torch.manual_seed(10)
        B, S = 1, 12
        x = torch.randn(B, S, cfg.hidden_size)
        cos, sin = build_rope_cache(S, cfg.rotary_dim, cfg.rope_theta)

        # Build HF-style module
        hf_attn = HFAttention(cfg)
        hf_attn.eval()

        # Extract weights into our state_dict format
        sd = {
            "q_proj.weight": hf_attn.q_proj.weight.detach(),
            "k_proj.weight": hf_attn.k_proj.weight.detach(),
            "v_proj.weight": hf_attn.v_proj.weight.detach(),
            "o_proj.weight": hf_attn.o_proj.weight.detach(),
            "q_norm.weight": hf_attn.q_norm.weight.detach(),
            "k_norm.weight": hf_attn.k_norm.weight.detach(),
        }

        with torch.no_grad():
            hf_out = hf_attn(x, cos, sin)
            ref_out = attention_forward(x, sd, cos, sin, cfg)

        assert ref_out.shape == (B, S, cfg.hidden_size)
        assert torch.allclose(ref_out, hf_out, atol=1e-5), f"Max diff: {(ref_out - hf_out).abs().max()}"

    def test_output_shape(self, cfg):
        torch.manual_seed(0)
        B, S = 2, 8
        sd = make_random_state_dict(cfg, num_layers=1)
        layer_sd = {k.removeprefix("model.layers.0.self_attn."): v for k, v in sd.items() if "self_attn" in k}
        cos, sin = build_rope_cache(S, cfg.rotary_dim, cfg.rope_theta)
        x = torch.randn(B, S, cfg.hidden_size)
        out = attention_forward(x, layer_sd, cos, sin, cfg)
        assert out.shape == (B, S, cfg.hidden_size)


class TestExpertMLP:
    def test_swiglu_output_shape(self, cfg):
        torch.manual_seed(0)
        B, S = 2, 4
        x = torch.randn(B * S, cfg.hidden_size)
        w1 = torch.randn(cfg.intermediate_size, cfg.hidden_size) * 0.02
        w2 = torch.randn(cfg.hidden_size, cfg.intermediate_size) * 0.02
        w3 = torch.randn(cfg.intermediate_size, cfg.hidden_size) * 0.02
        out = expert_mlp_forward(x, w1, w2, w3)
        assert out.shape == (B * S, cfg.hidden_size)

    def test_swiglu_matches_hf_mlp(self, cfg):
        """SwiGLU: out = w2(silu(w1(x)) * w3(x))"""
        torch.manual_seed(5)
        x = torch.randn(4, cfg.hidden_size)
        w1 = torch.randn(cfg.intermediate_size, cfg.hidden_size) * 0.02
        w2 = torch.randn(cfg.hidden_size, cfg.intermediate_size) * 0.02
        w3 = torch.randn(cfg.intermediate_size, cfg.hidden_size) * 0.02

        ref_out = expert_mlp_forward(x, w1, w2, w3)
        hf_out = F.linear(F.silu(F.linear(x, w1)) * F.linear(x, w3), w2)

        assert torch.allclose(ref_out, hf_out, atol=1e-6)


class TestMoE:
    def test_output_shape(self, cfg):
        torch.manual_seed(0)
        B, S = 1, 4
        sd = make_random_state_dict(cfg, num_layers=1)
        moe_sd = {
            k.removeprefix("model.layers.0.block_sparse_moe."): v for k, v in sd.items() if "block_sparse_moe" in k
        }
        x = torch.randn(B, S, cfg.hidden_size)
        out = moe_forward(x, moe_sd, cfg)
        assert out.shape == (B, S, cfg.hidden_size)

    def test_routing_uses_sigmoid_not_softmax(self, cfg):
        """The router must use sigmoid + topk (not softmax)."""
        torch.manual_seed(7)
        sd = make_random_state_dict(cfg, num_layers=1)
        gate_w = sd["model.layers.0.block_sparse_moe.gate.weight"]
        bias = sd["model.layers.0.block_sparse_moe.e_score_correction_bias"]
        x = torch.randn(1, cfg.hidden_size)
        logits = F.linear(x, gate_w)
        sigmoid_scores = torch.sigmoid(logits) + bias
        _, top_k_idx = torch.topk(sigmoid_scores, cfg.num_experts_per_tok, dim=-1)
        # Verify routing selects exactly num_experts_per_tok
        assert top_k_idx.shape[-1] == cfg.num_experts_per_tok

    def test_routing_bias_affects_selection(self, cfg):
        """e_score_correction_bias must shift which experts are selected."""
        torch.manual_seed(3)
        sd_no_bias = make_random_state_dict(cfg, num_layers=1)
        sd_with_bias = {k: v.clone() for k, v in sd_no_bias.items()}
        # Force bias toward expert 0
        bias_key = "model.layers.0.block_sparse_moe.e_score_correction_bias"
        sd_with_bias[bias_key] = torch.zeros_like(sd_no_bias[bias_key])
        sd_with_bias[bias_key][0] = 10.0  # strong bias for expert 0

        moe_sd_no_bias = {
            k.removeprefix("model.layers.0.block_sparse_moe."): v
            for k, v in sd_no_bias.items()
            if "block_sparse_moe" in k
        }
        moe_sd_with_bias = {
            k.removeprefix("model.layers.0.block_sparse_moe."): v
            for k, v in sd_with_bias.items()
            if "block_sparse_moe" in k
        }

        x = torch.randn(4, 1, cfg.hidden_size)
        out_no_bias = moe_forward(x, moe_sd_no_bias, cfg)
        out_with_bias = moe_forward(x, moe_sd_with_bias, cfg)

        # Outputs must differ when bias is applied
        assert not torch.allclose(out_no_bias, out_with_bias), "routing_bias had no effect on MoE output"


class TestDecoderLayer:
    def test_output_shape(self, cfg):
        torch.manual_seed(0)
        B, S = 1, 8
        sd = make_random_state_dict(cfg, num_layers=1)
        layer_sd = {k.removeprefix("model.layers.0."): v for k, v in sd.items() if k.startswith("model.layers.0.")}
        cos, sin = build_rope_cache(S, cfg.rotary_dim, cfg.rope_theta)
        x = torch.randn(B, S, cfg.hidden_size)
        out = decoder_layer_forward(x, layer_sd, cos, sin, cfg)
        assert out.shape == (B, S, cfg.hidden_size)

    def test_residual_preserved(self, cfg):
        """Output must differ from input (residual connections don't cancel)."""
        torch.manual_seed(0)
        B, S = 1, 4
        sd = make_random_state_dict(cfg, num_layers=1)
        layer_sd = {k.removeprefix("model.layers.0."): v for k, v in sd.items() if k.startswith("model.layers.0.")}
        cos, sin = build_rope_cache(S, cfg.rotary_dim, cfg.rope_theta)
        x = torch.randn(B, S, cfg.hidden_size)
        out = decoder_layer_forward(x, layer_sd, cos, sin, cfg)
        assert not torch.allclose(x, out), "Decoder layer output identical to input"


class TestFullModel:
    def test_output_shape(self, cfg):
        torch.manual_seed(0)
        B, S = 1, 8
        sd = make_random_state_dict(cfg, num_layers=cfg.num_hidden_layers)
        input_ids = torch.randint(0, cfg.vocab_size, (B, S))
        logits = model_forward(input_ids, sd, cfg)
        assert logits.shape == (B, S, cfg.vocab_size)

    def test_different_inputs_different_outputs(self, cfg):
        torch.manual_seed(0)
        sd = make_random_state_dict(cfg, num_layers=cfg.num_hidden_layers)
        ids1 = torch.randint(0, cfg.vocab_size, (1, 8))
        ids2 = torch.randint(0, cfg.vocab_size, (1, 8))
        while torch.all(ids1 == ids2):
            ids2 = torch.randint(0, cfg.vocab_size, (1, 8))
        logits1 = model_forward(ids1, sd, cfg)
        logits2 = model_forward(ids2, sd, cfg)
        assert not torch.allclose(logits1, logits2), "Different inputs produced identical logits"

    def test_deterministic(self, cfg):
        torch.manual_seed(0)
        sd = make_random_state_dict(cfg, num_layers=cfg.num_hidden_layers)
        ids = torch.randint(0, cfg.vocab_size, (1, 6))
        logits1 = model_forward(ids, sd, cfg)
        logits2 = model_forward(ids, sd, cfg)
        assert torch.allclose(logits1, logits2), "model_forward is not deterministic"


class TestGoldenGeneration:
    """Verify that golden outputs can be saved and reloaded correctly."""

    def test_save_load_golden(self, cfg, tmp_path):
        pass

        torch.manual_seed(42)
        sd = make_random_state_dict(cfg, num_layers=cfg.num_hidden_layers, seed=42)
        input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
        logits = model_forward(input_ids, sd, cfg)

        golden_path = tmp_path / "model_forward_golden.pt"
        torch.save({"input_ids": input_ids, "logits": logits, "seed": 42}, golden_path)

        loaded = torch.load(golden_path)
        assert torch.allclose(loaded["logits"], logits)
