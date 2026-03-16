# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MoE unit tests + comparison vs HF MoE module."""

import torch

from helpers import seed_all, assert_close, make_profiles
from moe import moe_torch, moe_tt, moe_shared_experts_torch, sigmoid_topk_routing


def _make_moe_weights(profile, dtype=torch.float32):
    h = profile["hidden_size"]
    ne = profile["num_experts"]
    mi = profile["moe_intermediate"]
    return dict(
        w_router=torch.randn(ne, h, dtype=dtype),
        w1=torch.randn(ne, mi, h, dtype=dtype),
        w2=torch.randn(ne, h, mi, dtype=dtype),
    )


def test_moe_matches_hf_per_profile_cpu_fp32():
    seed_all(42)
    profiles = make_profiles("tiny")
    b, seq = 2, 6
    for name, prof in profiles.items():
        if not prof["use_moe"]:
            continue
        h = prof["hidden_size"]
        x = torch.randn(b, seq, h)
        weights = _make_moe_weights(prof)
        out_torch = moe_torch(x, **weights, top_k=prof["top_k"])
        out_tt = moe_tt(x, **weights, top_k=prof["top_k"])
        assert_close(out_torch, out_tt, atol=1e-5, rtol=1e-5)


def test_moe_topk1():
    seed_all(99)
    prof = make_profiles("tiny")["glm4_355b"]
    prof = {**prof, "num_experts": 4, "top_k": 1, "use_moe": True}
    h = prof["hidden_size"]
    x = torch.randn(1, 1, h)
    weights = _make_moe_weights(prof)
    out_torch = moe_torch(x, **weights, top_k=1)
    out_tt = moe_tt(x, **weights, top_k=1)
    assert_close(out_torch, out_tt, atol=1e-5, rtol=1e-5)


def test_moe_batch1_seq1():
    seed_all(77)
    prof = make_profiles("tiny")["deepseek_v3"]
    h = prof["hidden_size"]
    x = torch.randn(1, 1, h)
    weights = _make_moe_weights(prof)
    out_torch = moe_torch(x, **weights, top_k=prof["top_k"])
    out_tt = moe_tt(x, **weights, top_k=prof["top_k"])
    assert_close(out_torch, out_tt, atol=1e-5, rtol=1e-5)


def test_moe_odd_seq():
    seed_all(55)
    prof = make_profiles("tiny")["gpt_oss_120b"]
    h = prof["hidden_size"]
    x = torch.randn(2, 7, h)  # odd seq length
    weights = _make_moe_weights(prof)
    out_torch = moe_torch(x, **weights, top_k=prof["top_k"])
    out_tt = moe_tt(x, **weights, top_k=prof["top_k"])
    assert_close(out_torch, out_tt, atol=1e-5, rtol=1e-5)


def _extract_swiglu_expert_weights(experts, ne):
    """Extract SwiGLU weights from HF expert modules → (w1, w2, w3)."""
    w1 = torch.stack([experts[e].gate_proj.weight.data for e in range(ne)])
    w3 = torch.stack([experts[e].up_proj.weight.data for e in range(ne)])
    w2 = torch.stack([experts[e].down_proj.weight.data for e in range(ne)])
    return w1, w2, w3


def _extract_shared_expert_weights(shared):
    """Extract shared expert SwiGLU weights from HF module."""
    return dict(
        w_gate=shared.gate_proj.weight.data,
        w_up=shared.up_proj.weight.data,
        w_down=shared.down_proj.weight.data,
    )


@torch.no_grad()
def test_moe_deepseek_v3():
    """
    Compose DeepSeek V3 MoE from building blocks (sigmoid_topk_routing +
    moe_forward + moe_shared_experts) and compare against HF module.
    """
    from transformers import DeepseekV3Config
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
        DeepseekV3DecoderLayer,
    )

    config = DeepseekV3Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_rope_head_dim=4,
        qk_nope_head_dim=8,
        v_head_dim=8,
        intermediate_size=128,
        moe_intermediate_size=32,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        first_k_dense_replace=0,
        n_shared_experts=1,
        num_hidden_layers=2,
        rms_norm_eps=1e-6,
        attn_implementation="eager",
    )

    seed_all(42)
    b, seq_len = 1, 4
    x = torch.randn(b, seq_len, config.hidden_size)

    layer = DeepseekV3DecoderLayer(config, layer_idx=0)
    layer.eval()

    # 1) HF reference
    moe_mod = layer.mlp
    hf_out = moe_mod(x)

    # 2) Extract weights
    ne = config.n_routed_experts
    w1, w2, w3 = _extract_swiglu_expert_weights(moe_mod.experts, ne)
    shared_kwargs = _extract_shared_expert_weights(moe_mod.shared_experts)

    # 3) Compose from building blocks: routing → routed experts → shared experts
    topk_indices, topk_weights = sigmoid_topk_routing(
        x,
        w_router=moe_mod.gate.weight.data,
        top_k=config.num_experts_per_tok,
        n_group=config.n_group,
        topk_group=config.topk_group,
        e_score_correction_bias=moe_mod.gate.e_score_correction_bias,
        routed_scaling_factor=config.routed_scaling_factor,
        norm_topk_prob=config.norm_topk_prob,
    )

    our_out = moe_torch(
        x, w_router=moe_mod.gate.weight.data, w1=w1, w2=w2, w3=w3,
        top_k=config.num_experts_per_tok,
        precomputed_routing=(topk_indices, topk_weights),
    )
    our_out = our_out + moe_shared_experts_torch(x, **shared_kwargs)

    # 4) Compare
    assert_close(hf_out, our_out, atol=1e-5, rtol=1e-5), \
        "moe building blocks vs HF DeepSeek V3 MoE mismatch"
