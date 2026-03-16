"""Full decoder tests: per-family forward comparison + compose-from-primitives vs HF reference."""

import torch
import torch.nn.functional as F
import pytest

from helpers import seed_all, assert_close, make_profiles, rms_norm
from gqa import gqa_attention_torch, gqa_attention_tt
from mla import mla_attention_torch, mla_attention_tt
from moe import moe_torch, moe_tt, moe_shared_experts_torch, moe_shared_experts_tt, sigmoid_topk_routing

# ---------------------------------------------------------------------------
# Dispatch: variant string → function implementations
# ---------------------------------------------------------------------------

IMPL = {
    "torch": dict(
        gqa=gqa_attention_torch,
        mla=mla_attention_torch,
        moe=moe_torch,
        shared=moe_shared_experts_torch,
    ),
    "tt": dict(
        gqa=gqa_attention_tt,
        mla=mla_attention_tt,
        moe=moe_tt,
        shared=moe_shared_experts_tt,
    ),
}


def fns(variant):
    """Return (gqa_fn, mla_fn, moe_fn, shared_fn) for the given variant."""
    d = IMPL[variant]
    return d["gqa"], d["mla"], d["moe"], d["shared"]


# ---------------------------------------------------------------------------
# Weight construction
# ---------------------------------------------------------------------------

def _make_decoder_weights(prof, dtype=torch.float32):
    h = prof["hidden_size"]
    nq = prof["num_q_heads"]
    nkv = prof["num_kv_heads"]
    hd = prof["head_dim"]
    ffn_i = prof["ffn_intermediate"]

    w = {}

    # Layer norms
    w["ln1_weight"] = torch.ones(h, dtype=dtype)
    w["ln1_bias"] = torch.zeros(h, dtype=dtype)
    w["ln2_weight"] = torch.ones(h, dtype=dtype)
    w["ln2_bias"] = torch.zeros(h, dtype=dtype)

    # Attention weights
    if prof["use_mla"]:
        lat = prof["kv_latent_dim"]
        w["attn"] = dict(
            wq=torch.randn(nq * hd, h, dtype=dtype),
            w_kv_down=torch.randn(lat, h, dtype=dtype),
            w_k_up=torch.randn(nkv * hd, lat, dtype=dtype),
            w_v_up=torch.randn(nkv * hd, lat, dtype=dtype),
            wo=torch.randn(h, nq * hd, dtype=dtype),
        )
    else:
        w["attn"] = dict(
            wq=torch.randn(nq * hd, h, dtype=dtype),
            wk=torch.randn(nkv * hd, h, dtype=dtype),
            wv=torch.randn(nkv * hd, h, dtype=dtype),
            wo=torch.randn(h, nq * hd, dtype=dtype),
        )

    # FFN weights
    if prof["use_moe"]:
        ne = prof["num_experts"]
        mi = prof["moe_intermediate"]
        w["ffn"] = dict(
            w_router=torch.randn(ne, h, dtype=dtype),
            w1=torch.randn(ne, mi, h, dtype=dtype),
            w2=torch.randn(ne, h, mi, dtype=dtype),
        )
    else:
        w["ffn"] = dict(
            w1=torch.randn(ffn_i, h, dtype=dtype),
            w2=torch.randn(h, ffn_i, dtype=dtype),
        )

    return w


# ---------------------------------------------------------------------------
# Decoder forward
# ---------------------------------------------------------------------------

def decoder_forward(hidden_states, profile, weights, variant="torch",
                    attention_mask=None, past_key_value=None, use_cache=False):
    """
    Minimal decoder block: pre-norm -> attention -> residual -> pre-norm -> FFN -> residual.
    """
    gqa_fn, mla_fn, moe_fn, _ = fns(variant)
    h = hidden_states

    # Pre-norm + attention
    normed = F.layer_norm(h, [profile["hidden_size"]],
                          weight=weights["ln1_weight"], bias=weights["ln1_bias"])

    attn_kwargs = dict(
        num_q_heads=profile["num_q_heads"],
        num_kv_heads=profile["num_kv_heads"],
        attention_mask=attention_mask,
        past_key_value=past_key_value,
        use_cache=use_cache,
    )

    if profile["use_mla"]:
        attn_out, present_kv = mla_fn(
            normed, **weights["attn"],
            kv_latent_dim=profile["kv_latent_dim"],
            **attn_kwargs,
        )
    else:
        attn_out, present_kv = gqa_fn(normed, **weights["attn"], **attn_kwargs)

    h = h + attn_out

    # Pre-norm + FFN
    normed = F.layer_norm(h, [profile["hidden_size"]],
                          weight=weights["ln2_weight"], bias=weights["ln2_bias"])

    if profile["use_moe"]:
        ffn_out = moe_fn(normed, **weights["ffn"], top_k=profile["top_k"])
    else:
        # Dense 2-linear MLP
        ffn_out = F.silu(F.linear(normed, weights["ffn"]["w1"]))
        ffn_out = F.linear(ffn_out, weights["ffn"]["w2"])

    h = h + ffn_out
    return h, present_kv


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_decoder_matches_hf_per_family():
    seed_all(42)
    profiles = make_profiles("tiny")
    b, seq = 2, 6
    for name, prof in profiles.items():
        h = prof["hidden_size"]
        x = torch.randn(b, seq, h)
        weights = _make_decoder_weights(prof)

        out_torch, _ = decoder_forward(x, prof, weights, variant="torch")
        out_tt, _ = decoder_forward(x, prof, weights, variant="tt")
        assert_close(out_torch, out_tt, atol=1e-5, rtol=1e-5), f"Failed for {name}"


def test_decoder_cache_chunking_equivalence_per_family():
    seed_all(123)
    profiles = make_profiles("tiny")
    b, seq = 1, 8
    split = 4

    for name, prof in profiles.items():
        h = prof["hidden_size"]
        x = torch.randn(b, seq, h)
        weights = _make_decoder_weights(prof)

        # Full forward with causal mask
        causal_full = torch.tril(torch.ones(seq, seq, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        out_full, _ = decoder_forward(x, prof, weights, attention_mask=causal_full)

        # Chunk 1
        x1 = x[:, :split, :]
        causal1 = torch.tril(torch.ones(split, split, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        out1, cache = decoder_forward(
            x1, prof, weights, attention_mask=causal1, use_cache=True,
        )

        # Chunk 2
        x2 = x[:, split:, :]
        kv_len = seq
        causal2 = torch.ones(seq - split, kv_len, dtype=torch.bool)
        for i in range(seq - split):
            causal2[i, split + i + 1:] = False
        causal2 = causal2.unsqueeze(0).unsqueeze(0)
        out2, _ = decoder_forward(
            x2, prof, weights, attention_mask=causal2,
            past_key_value=cache, use_cache=True,
        )

        out_chunked = torch.cat([out1, out2], dim=1)
        assert_close(out_full, out_chunked, atol=1e-2, rtol=1e-2), f"Cache chunking failed for {name}"


# ===========================================================================
# Real HF model comparison tests
# ===========================================================================

# --- Helpers for manual forward replication ---



# ---------------------------------------------------------------------------
# Full DeepSeek V3 decoder layer composed from mla.py + moe.py functions
# ---------------------------------------------------------------------------

def _compose_deepseek_v3_forward(x, params, variant):
    """
    Compose a DeepSeek V3 decoder layer from primitives (MLA + MoE/dense FFN).
    params: dict with all weights/config needed for the forward pass.
    """
    _, mla_fn, moe_fn, shared_fn = fns(variant)

    h = x

    # Input RMSNorm → MLA attention → residual
    normed = rms_norm(h, params["ln1_weight"], params["eps"])
    mla_kwargs = {k: params[k] for k in (
        "wq", "w_kv_down", "w_k_up", "w_v_up", "wo",
        "num_q_heads", "num_kv_heads", "kv_latent_dim",
    )}
    for k in ("position_embeddings", "qk_rope_head_dim", "rope_interleave"):
        if k in params:
            mla_kwargs[k] = params[k]
    attn_out, _ = mla_fn(normed, **mla_kwargs)
    h = h + attn_out

    # Post-attention RMSNorm → FFN → residual
    normed2 = rms_norm(h, params["ln2_weight"], params["eps"])

    if params["is_dense"]:
        ffn_out = shared_fn(
            normed2,
            w_gate=params["dense_gate"], w_up=params["dense_up"],
            w_down=params["dense_down"],
        )
    else:
        topk_indices, topk_weights = sigmoid_topk_routing(
            normed2,
            w_router=params["moe_router"],
            top_k=params["top_k"],
            n_group=params["n_group"],
            topk_group=params["topk_group"],
            e_score_correction_bias=params.get("e_score_correction_bias"),
            routed_scaling_factor=params.get("routed_scaling_factor", 1.0),
            norm_topk_prob=params.get("norm_topk_prob", False),
        )
        ffn_out = moe_fn(
            normed2,
            w_router=params["moe_router"], w1=params["moe_w1"],
            w2=params["moe_w2"], w3=params["moe_w3"],
            top_k=params["top_k"],
            precomputed_routing=(topk_indices, topk_weights),
        )
        ffn_out = ffn_out + shared_fn(
            normed2,
            w_gate=params["shared_gate"], w_up=params["shared_up"],
            w_down=params["shared_down"],
        )

    return h + ffn_out


@torch.no_grad()
@pytest.mark.parametrize("variant", IMPL.keys())
def test_deepseek_v3_decoder_from_primitives(variant):
    """
    Compose a full DeepSeek V3 decoder layer from our primitive functions
    (mla_attention_torch/hf from mla.py, moe_torch/hf + moe_shared_experts
    from moe.py) and compare against the real HF DeepseekV3DecoderLayer.

    Tests both dense (layer 0) and MoE (layer 1) decoder layers.
    """
    import torch.nn as nn
    from transformers import DeepseekV3Config
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
        DeepseekV3DecoderLayer,
        DeepseekV3RotaryEmbedding,
    )

    nope_dim = 8
    rope_dim = 4
    v_head_dim = 8
    config = DeepseekV3Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=None,
        kv_lora_rank=16,
        qk_rope_head_dim=rope_dim,
        qk_nope_head_dim=nope_dim,
        v_head_dim=v_head_dim,
        intermediate_size=128,
        moe_intermediate_size=32,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        first_k_dense_replace=1,  # layer 0 = dense, layer 1 = MoE
        n_shared_experts=1,
        num_hidden_layers=2,
        attention_bias=False,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        rope_theta=10000.0,
        attn_implementation="eager",
    )

    seed_all(42)
    b, seq_len = 1, 4
    x = torch.randn(b, seq_len, config.hidden_size)
    num_heads = config.num_attention_heads
    rotary = DeepseekV3RotaryEmbedding(config)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary(x, position_ids)
    label = variant

    for layer_idx in [0, 1]:
        layer = DeepseekV3DecoderLayer(config, layer_idx=layer_idx)
        layer.eval()
        layer.self_attn.kv_a_layernorm = nn.Identity()

        # 1) HF reference
        hf_out = layer(x, position_embeddings=(cos, sin))

        # 2) Collect params/weights
        attn = layer.self_attn
        kv_b_w = attn.kv_b_proj.weight.data
        kv_b_reshaped = kv_b_w.view(num_heads, nope_dim + v_head_dim, config.kv_lora_rank)
        w_k_up = kv_b_reshaped[:, :nope_dim, :].reshape(num_heads * nope_dim, config.kv_lora_rank)
        w_v_up = kv_b_reshaped[:, nope_dim:, :].reshape(num_heads * v_head_dim, config.kv_lora_rank)

        is_dense = layer_idx < config.first_k_dense_replace
        params = dict(
            eps=config.rms_norm_eps,
            ln1_weight=layer.input_layernorm.weight.data,
            ln2_weight=layer.post_attention_layernorm.weight.data,
            wq=attn.q_proj.weight.data,
            w_kv_down=attn.kv_a_proj_with_mqa.weight.data,
            w_k_up=w_k_up, w_v_up=w_v_up,
            wo=attn.o_proj.weight.data,
            num_q_heads=num_heads,
            num_kv_heads=config.num_key_value_heads,
            kv_latent_dim=config.kv_lora_rank,
            position_embeddings=(cos, sin),
            qk_rope_head_dim=rope_dim,
            rope_interleave=True,
            is_dense=is_dense,
        )

        if is_dense:
            mlp = layer.mlp
            params.update(
                dense_gate=mlp.gate_proj.weight.data,
                dense_up=mlp.up_proj.weight.data,
                dense_down=mlp.down_proj.weight.data,
            )
        else:
            moe_mod = layer.mlp
            ne = config.n_routed_experts
            params.update(
                moe_router=moe_mod.gate.weight.data,
                moe_w1=torch.stack([moe_mod.experts[e].gate_proj.weight.data for e in range(ne)]),
                moe_w3=torch.stack([moe_mod.experts[e].up_proj.weight.data for e in range(ne)]),
                moe_w2=torch.stack([moe_mod.experts[e].down_proj.weight.data for e in range(ne)]),
                top_k=config.num_experts_per_tok,
                n_group=config.n_group,
                topk_group=config.topk_group,
                e_score_correction_bias=moe_mod.gate.e_score_correction_bias,
                routed_scaling_factor=config.routed_scaling_factor,
                norm_topk_prob=config.norm_topk_prob,
                shared_gate=moe_mod.shared_experts.gate_proj.weight.data,
                shared_up=moe_mod.shared_experts.up_proj.weight.data,
                shared_down=moe_mod.shared_experts.down_proj.weight.data,
            )

        # 3) Run composed forward
        h = _compose_deepseek_v3_forward(x, params, variant)

        # 4) Compare
        assert_close(hf_out, h, atol=1e-5, rtol=1e-5), (
            f"DeepSeek V3 ({label}): composed vs real HF "
            f"layer_idx={layer_idx} mismatch"
        )


# ---------------------------------------------------------------------------
# Full Kimi K2.5 decoder layer composed from mla.py + moe.py functions
# ---------------------------------------------------------------------------

@torch.no_grad()
@pytest.mark.parametrize("variant", IMPL.keys())
def test_kimi_k25_decoder_from_primitives(variant):
    """Compose Kimi K2.5 decoder (DeepSeek V3 architecture with different params)."""
    import torch.nn as nn
    from transformers import DeepseekV3Config
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
        DeepseekV3DecoderLayer,
        DeepseekV3RotaryEmbedding,
    )

    nope_dim = 8
    rope_dim = 4
    v_head_dim = 8
    config = DeepseekV3Config(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=4,
        q_lora_rank=None, kv_lora_rank=16, qk_rope_head_dim=rope_dim,
        qk_nope_head_dim=nope_dim, v_head_dim=v_head_dim,
        intermediate_size=128, moe_intermediate_size=32,
        n_routed_experts=4, num_experts_per_tok=2,
        n_group=1, topk_group=1, routed_scaling_factor=2.827,
        first_k_dense_replace=1, n_shared_experts=1,
        num_hidden_layers=2, attention_bias=False,
        rms_norm_eps=1e-5, rope_theta=50000.0,
        max_position_embeddings=256, attn_implementation="eager",
    )

    seed_all(77)
    b, seq_len = 1, 4
    x = torch.randn(b, seq_len, config.hidden_size)
    num_heads = config.num_attention_heads
    rotary = DeepseekV3RotaryEmbedding(config)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary(x, position_ids)
    label = variant

    for layer_idx in [0, 1]:
        layer = DeepseekV3DecoderLayer(config, layer_idx=layer_idx)
        layer.eval()
        layer.self_attn.kv_a_layernorm = nn.Identity()

        # 1) HF reference
        hf_out = layer(x, position_embeddings=(cos, sin))

        # 2) Collect params
        attn = layer.self_attn
        kv_b_w = attn.kv_b_proj.weight.data
        kv_b_reshaped = kv_b_w.view(num_heads, nope_dim + v_head_dim, config.kv_lora_rank)
        w_k_up = kv_b_reshaped[:, :nope_dim, :].reshape(num_heads * nope_dim, config.kv_lora_rank)
        w_v_up = kv_b_reshaped[:, nope_dim:, :].reshape(num_heads * v_head_dim, config.kv_lora_rank)

        is_dense = layer_idx < config.first_k_dense_replace
        params = dict(
            eps=config.rms_norm_eps,
            ln1_weight=layer.input_layernorm.weight.data,
            ln2_weight=layer.post_attention_layernorm.weight.data,
            wq=attn.q_proj.weight.data,
            w_kv_down=attn.kv_a_proj_with_mqa.weight.data,
            w_k_up=w_k_up, w_v_up=w_v_up,
            wo=attn.o_proj.weight.data,
            num_q_heads=num_heads,
            num_kv_heads=config.num_key_value_heads,
            kv_latent_dim=config.kv_lora_rank,
            position_embeddings=(cos, sin),
            qk_rope_head_dim=rope_dim,
            rope_interleave=True,
            is_dense=is_dense,
        )

        if is_dense:
            mlp = layer.mlp
            params.update(
                dense_gate=mlp.gate_proj.weight.data,
                dense_up=mlp.up_proj.weight.data,
                dense_down=mlp.down_proj.weight.data,
            )
        else:
            moe_mod = layer.mlp
            ne = config.n_routed_experts
            params.update(
                moe_router=moe_mod.gate.weight.data,
                moe_w1=torch.stack([moe_mod.experts[e].gate_proj.weight.data for e in range(ne)]),
                moe_w3=torch.stack([moe_mod.experts[e].up_proj.weight.data for e in range(ne)]),
                moe_w2=torch.stack([moe_mod.experts[e].down_proj.weight.data for e in range(ne)]),
                top_k=config.num_experts_per_tok,
                n_group=config.n_group,
                topk_group=config.topk_group,
                e_score_correction_bias=moe_mod.gate.e_score_correction_bias,
                routed_scaling_factor=config.routed_scaling_factor,
                norm_topk_prob=config.norm_topk_prob,
                shared_gate=moe_mod.shared_experts.gate_proj.weight.data,
                shared_up=moe_mod.shared_experts.up_proj.weight.data,
                shared_down=moe_mod.shared_experts.down_proj.weight.data,
            )

        # 3) Run composed forward
        h = _compose_deepseek_v3_forward(x, params, variant)

        # 4) Compare
        assert_close(hf_out, h, atol=1e-4, rtol=1e-4), (
            f"Kimi K2.5 ({label}): composed vs real HF "
            f"layer_idx={layer_idx} mismatch"
        )


# ---------------------------------------------------------------------------
# Full GLM-4 decoder layer composed from gqa.py + moe.py functions
# ---------------------------------------------------------------------------

@torch.no_grad()
@pytest.mark.parametrize("variant", IMPL.keys())
def test_glm4_decoder_from_primitives(variant):
    """
    Compose a full GLM-4 decoder layer from our primitive functions.

    GLM-4 structure (4x RMSNorm):
    1) input_layernorm → attention
    2) post_self_attn_layernorm(attn_out) → residual
    3) post_attention_layernorm → SwiGLU MLP
    4) post_mlp_layernorm(mlp_out) → residual
    """
    from transformers import Glm4Config
    from transformers.models.glm4.modeling_glm4 import (
        Glm4DecoderLayer,
        Glm4RotaryEmbedding,
    )

    config = Glm4Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        partial_rotary_factor=0.5,
        attention_bias=False,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        rope_theta=10000.0,
        num_hidden_layers=2,
        attn_implementation="eager",
    )

    seed_all(42)
    b, seq_len = 1, 4
    x = torch.randn(b, seq_len, config.hidden_size)

    layer = Glm4DecoderLayer(config, layer_idx=0)
    layer.eval()

    rotary = Glm4RotaryEmbedding(config)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary(x, position_ids)

    # 1) HF reference
    hf_out = layer(x, position_embeddings=(cos, sin))

    # 2) Collect params (GLM-4 has fused gate_up_proj and 4 RMSNorms)
    attn = layer.self_attn
    mlp = layer.mlp
    intermediate = config.intermediate_size
    gate_up_w = mlp.gate_up_proj.weight.data
    params = dict(
        eps=config.rms_norm_eps,
        ln1_weight=layer.input_layernorm.weight.data,
        ln_post_attn=layer.post_self_attn_layernorm.weight.data,
        ln2_weight=layer.post_attention_layernorm.weight.data,
        ln_post_mlp=layer.post_mlp_layernorm.weight.data,
        gqa_kwargs=dict(
            wq=attn.q_proj.weight.data, wk=attn.k_proj.weight.data,
            wv=attn.v_proj.weight.data, wo=attn.o_proj.weight.data,
            num_q_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            position_embeddings=(cos, sin),
            rope_variant="glm4",
        ),
        dense_gate=gate_up_w[:intermediate, :],
        dense_up=gate_up_w[intermediate:, :],
        dense_down=mlp.down_proj.weight.data,
    )

    # 3) Run composed forward
    h = _compose_glm4_forward(x, params, variant)

    # 4) Compare
    label = variant
    assert_close(hf_out, h, atol=1e-4, rtol=1e-4), \
        f"GLM-4 ({label}): composed vs real HF GLM-4 mismatch"


# ---------------------------------------------------------------------------
# Full GPT-OSS 120B decoder layer composed from gqa.py + moe.py functions
#   (using Mixtral as real HF reference — same GQA + MoE architecture)
# ---------------------------------------------------------------------------

@torch.no_grad()
@pytest.mark.parametrize("variant", IMPL.keys())
def test_gpt_oss_decoder_from_primitives(variant):
    """
    GPT-OSS 120B uses GQA + MoE (Mixtral as HF proxy).
    """
    from transformers import MixtralConfig
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralDecoderLayer,
        MixtralRotaryEmbedding,
    )

    config = MixtralConfig(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, intermediate_size=128, num_local_experts=4,
        num_experts_per_tok=2, rms_norm_eps=1e-6, max_position_embeddings=256,
        rope_theta=10000.0, num_hidden_layers=2, router_jitter_noise=0.0,
        attn_implementation="eager",
    )

    seed_all(42)
    b, seq_len = 1, 4
    x = torch.randn(b, seq_len, config.hidden_size)

    layer = MixtralDecoderLayer(config, layer_idx=0)
    layer.eval()

    rotary = MixtralRotaryEmbedding(config)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary(x, position_ids)

    # 1) HF reference
    hf_out = layer(x, position_embeddings=(cos, sin))

    # 2) Collect params
    attn = layer.self_attn
    moe_mod = layer.block_sparse_moe
    ne = config.num_local_experts
    w1, w2, w3 = _extract_mixtral_expert_weights(moe_mod.experts, ne)
    params = dict(
        eps=config.rms_norm_eps,
        ln1_weight=layer.input_layernorm.weight.data,
        ln2_weight=layer.post_attention_layernorm.weight.data,
        gqa_kwargs=dict(
            wq=attn.q_proj.weight.data, wk=attn.k_proj.weight.data,
            wv=attn.v_proj.weight.data, wo=attn.o_proj.weight.data,
            num_q_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            position_embeddings=(cos, sin),
        ),
        moe_router=moe_mod.gate.weight.data,
        moe_w1=w1, moe_w2=w2, moe_w3=w3,
        top_k=config.num_experts_per_tok,
    )

    # 3) Run composed forward
    h = _compose_gqa_moe_forward(x, params, variant)

    # 4) Compare
    label = variant
    assert_close(hf_out, h, atol=1e-4, rtol=1e-4), \
        f"GPT-OSS 120B ({label}): composed vs real HF Mixtral proxy mismatch"


# ===========================================================================
# Helpers for composed decoder tests
# ===========================================================================

def _softmax_topk_routing(normed, gate_weight, top_k, normalize=True):
    """Extract softmax→topk→renormalize routing (Mixtral/Qwen3Moe/OLMoE pattern)."""
    flat = normed.reshape(-1, normed.shape[-1])
    router_logits = F.linear(flat, gate_weight)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, topk_indices = torch.topk(routing_weights, top_k, dim=-1)
    if normalize:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(flat.dtype)
    return topk_indices, routing_weights


def _extract_swiglu_expert_weights(experts, ne, attr_gate="gate_proj", attr_up="up_proj", attr_down="down_proj"):
    """Extract SwiGLU weights from HF expert modules (gate/up/down naming)."""
    w1 = torch.stack([getattr(experts[e], attr_gate).weight.data for e in range(ne)])
    w3 = torch.stack([getattr(experts[e], attr_up).weight.data for e in range(ne)])
    w2 = torch.stack([getattr(experts[e], attr_down).weight.data for e in range(ne)])
    return w1, w2, w3


def _extract_mixtral_expert_weights(experts, ne):
    """Extract SwiGLU weights from Mixtral-style experts (w1/w3/w2 naming)."""
    w1 = torch.stack([experts[e].w1.weight.data for e in range(ne)])
    w3 = torch.stack([experts[e].w3.weight.data for e in range(ne)])
    w2 = torch.stack([experts[e].w2.weight.data for e in range(ne)])
    return w1, w2, w3


def _collect_gqa_dense_params(layer, config, position_embeddings=None, rope_variant="standard"):
    """Extract params dict from a Llama/Qwen-style GQA + dense SwiGLU layer."""
    attn = layer.self_attn
    mlp = layer.mlp
    gqa_kwargs = dict(
        wq=attn.q_proj.weight.data, wk=attn.k_proj.weight.data,
        wv=attn.v_proj.weight.data, wo=attn.o_proj.weight.data,
        num_q_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
    )
    if position_embeddings is not None:
        gqa_kwargs["position_embeddings"] = position_embeddings
        gqa_kwargs["rope_variant"] = rope_variant
    return dict(
        eps=config.rms_norm_eps,
        ln1_weight=layer.input_layernorm.weight.data,
        ln2_weight=layer.post_attention_layernorm.weight.data,
        gqa_kwargs=gqa_kwargs,
        dense_gate=mlp.gate_proj.weight.data,
        dense_up=mlp.up_proj.weight.data,
        dense_down=mlp.down_proj.weight.data,
    )


def _compose_gqa_dense_forward(x, params, variant):
    """Compose GQA + dense SwiGLU decoder (Llama/Qwen2 pattern, 2 RMSNorms)."""
    gqa_fn, _, _, shared_fn = fns(variant)

    h = x
    normed = rms_norm(h, params["ln1_weight"], params["eps"])
    attn_out, _ = gqa_fn(normed, **params["gqa_kwargs"])
    h = h + attn_out

    normed2 = rms_norm(h, params["ln2_weight"], params["eps"])
    ffn_out = shared_fn(
        normed2,
        w_gate=params["dense_gate"], w_up=params["dense_up"],
        w_down=params["dense_down"],
    )
    return h + ffn_out


def _compose_glm4_forward(x, params, variant):
    """Compose GLM-4 decoder (GQA + dense SwiGLU, 4 RMSNorms with post-attn/post-mlp norms)."""
    gqa_fn, _, _, shared_fn = fns(variant)

    h = x
    normed = rms_norm(h, params["ln1_weight"], params["eps"])
    attn_out, _ = gqa_fn(normed, **params["gqa_kwargs"])
    attn_out = rms_norm(attn_out, params["ln_post_attn"], params["eps"])
    h = h + attn_out

    normed2 = rms_norm(h, params["ln2_weight"], params["eps"])
    mlp_out = shared_fn(
        normed2,
        w_gate=params["dense_gate"], w_up=params["dense_up"],
        w_down=params["dense_down"],
    )
    mlp_out = rms_norm(mlp_out, params["ln_post_mlp"], params["eps"])
    return h + mlp_out


def _compose_gqa_moe_forward(x, params, variant):
    """Compose GQA + softmax MoE decoder (Mixtral/OLMoE/Qwen3Moe pattern, 2 RMSNorms)."""
    gqa_fn, _, moe_fn, _ = fns(variant)

    h = x
    normed = rms_norm(h, params["ln1_weight"], params["eps"])
    attn_out, _ = gqa_fn(normed, **params["gqa_kwargs"])
    h = h + attn_out

    normed2 = rms_norm(h, params["ln2_weight"], params["eps"])
    topk_idx, rw = _softmax_topk_routing(
        normed2, params["moe_router"], params["top_k"],
        normalize=params.get("normalize_routing", True),
    )
    ffn_out = moe_fn(
        normed2, w_router=params["moe_router"],
        w1=params["moe_w1"], w2=params["moe_w2"], w3=params["moe_w3"],
        top_k=params["top_k"],
        precomputed_routing=(topk_idx, rw),
    )
    return h + ffn_out


def _compose_minimax_forward(x, params, variant):
    """Compose MiniMax decoder (GQA + MoE with weighted residual)."""
    gqa_fn, _, moe_fn, _ = fns(variant)

    normed = rms_norm(x, params["ln1_weight"], params["eps"])
    attn_out, _ = gqa_fn(normed, **params["gqa_kwargs"])
    h = normed * params["alpha_attn"] + attn_out * params["beta_attn"]

    normed2 = rms_norm(h, params["ln2_weight"], params["eps"])
    topk_idx, rw = _softmax_topk_routing(
        normed2, params["moe_router"], params["top_k"])
    moe_out = moe_fn(
        normed2, w_router=params["moe_router"],
        w1=params["moe_w1"], w2=params["moe_w2"], w3=params["moe_w3"],
        top_k=params["top_k"],
        precomputed_routing=(topk_idx, rw),
    )
    return normed2 * params["alpha_mlp"] + moe_out * params["beta_mlp"]


def _compose_llama4_forward(x, params, variant):
    """Compose Llama4 decoder (GQA + sigmoid MoE with shared expert, scale_input)."""
    gqa_fn, _, moe_fn, shared_fn = fns(variant)

    h = x
    normed = rms_norm(h, params["ln1_weight"], params["eps"])
    attn_out, _ = gqa_fn(normed, **params["gqa_kwargs"])
    h = h + attn_out

    normed2 = rms_norm(h, params["ln2_weight"], params["eps"])

    # Llama4 sigmoid routing: topk → sigmoid
    flat = normed2.reshape(-1, normed2.shape[-1])
    router_logits = F.linear(flat, params["moe_router"])
    router_top_value, topk_indices = torch.topk(
        router_logits, params["top_k"], dim=1)
    routing_weights = torch.sigmoid(router_top_value.float()).to(flat.dtype)

    ffn_out = moe_fn(
        normed2, w_router=params["moe_router"],
        w1=params["moe_w1"], w2=params["moe_w2"], w3=params["moe_w3"],
        top_k=params["top_k"],
        precomputed_routing=(topk_indices, routing_weights),
        scale_input=True,
    )
    ffn_out = ffn_out + shared_fn(
        normed2,
        w_gate=params["shared_gate"], w_up=params["shared_up"],
        w_down=params["shared_down"],
    )
    return h + ffn_out


# ===========================================================================
# Llama-architecture decoders (GQA + dense SwiGLU, 2 RMSNorms)
#   GPT-OSS 20B, GPT-OSS Safeguard 20B, Llama Guard 4, Llama 3.1, Llama 3.3
# ===========================================================================

@torch.no_grad()
@pytest.mark.parametrize("variant", IMPL.keys())
@pytest.mark.parametrize("model_name,config_overrides,seed", [
    ("GPT-OSS 20B", None, 42),
    ("GPT-OSS Safeguard 20B", dict(rope_theta=500000.0), 43),
    ("Llama Guard 4 12B", dict(rope_theta=500000.0), 44),
    ("Llama 3.1 8B", dict(rope_theta=500000.0), 45),
    ("Llama 3.3 70B", dict(num_key_value_heads=4, rope_theta=500000.0), 46),
], ids=["gpt_oss_20b", "gpt_oss_safeguard_20b", "llama_guard4", "llama3_1_8b", "llama3_3_70b"])
def test_llama_decoder_from_primitives(variant, model_name, config_overrides, seed):
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import (
        LlamaDecoderLayer,
        LlamaRotaryEmbedding,
    )

    base_kwargs = dict(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, intermediate_size=128, rms_norm_eps=1e-6,
        max_position_embeddings=256, rope_theta=10000.0,
        num_hidden_layers=2, attn_implementation="eager",
    )
    if config_overrides:
        base_kwargs.update(config_overrides)

    config = LlamaConfig(**base_kwargs)
    seed_all(seed)
    b, seq_len = 1, 4
    x = torch.randn(b, seq_len, config.hidden_size)

    layer = LlamaDecoderLayer(config, layer_idx=0)
    layer.eval()

    rotary = LlamaRotaryEmbedding(config)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary(x, position_ids)

    # 1) HF reference
    hf_out = layer(x, position_embeddings=(cos, sin))

    # 2) Collect params
    params = _collect_gqa_dense_params(layer, config, position_embeddings=(cos, sin))

    # 3) Run composed forward
    h = _compose_gqa_dense_forward(x, params, variant)

    # 4) Compare
    label = variant
    assert_close(hf_out, h, atol=1e-4, rtol=1e-4), \
        f"{model_name} ({label}): composed vs real HF Llama mismatch"


@torch.no_grad()
@pytest.mark.parametrize("variant", IMPL.keys())
def test_grok2_decoder_from_primitives(variant):
    """Grok 2 270B uses GQA + MoE (Mixtral as HF proxy)."""
    from transformers import MixtralConfig
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralDecoderLayer,
        MixtralRotaryEmbedding,
    )

    config = MixtralConfig(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, intermediate_size=128, num_local_experts=8,
        num_experts_per_tok=2, rms_norm_eps=1e-6, max_position_embeddings=256,
        rope_theta=10000.0, num_hidden_layers=2, router_jitter_noise=0.0,
        attn_implementation="eager",
    )

    seed_all(55)
    b, seq_len = 1, 4
    x = torch.randn(b, seq_len, config.hidden_size)

    layer = MixtralDecoderLayer(config, layer_idx=0)
    layer.eval()

    rotary = MixtralRotaryEmbedding(config)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary(x, position_ids)

    # 1) HF reference
    hf_out = layer(x, position_embeddings=(cos, sin))

    # 2) Collect params
    attn = layer.self_attn
    moe_mod = layer.block_sparse_moe
    w1, w2, w3 = _extract_mixtral_expert_weights(moe_mod.experts, config.num_local_experts)
    params = dict(
        eps=config.rms_norm_eps,
        ln1_weight=layer.input_layernorm.weight.data,
        ln2_weight=layer.post_attention_layernorm.weight.data,
        gqa_kwargs=dict(
            wq=attn.q_proj.weight.data, wk=attn.k_proj.weight.data,
            wv=attn.v_proj.weight.data, wo=attn.o_proj.weight.data,
            num_q_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            position_embeddings=(cos, sin),
        ),
        moe_router=moe_mod.gate.weight.data,
        moe_w1=w1, moe_w2=w2, moe_w3=w3,
        top_k=config.num_experts_per_tok,
    )

    # 3) Run composed forward
    h = _compose_gqa_moe_forward(x, params, variant)

    # 4) Compare
    label = variant
    assert_close(hf_out, h, atol=1e-4, rtol=1e-4), \
        f"Grok 2 ({label}): composed vs real HF Mixtral proxy mismatch"


# ---------------------------------------------------------------------------
# Llama4 Maverick / Scout — GQA + MoE (sigmoid routing, shared expert)
# ---------------------------------------------------------------------------

@torch.no_grad()
@pytest.mark.parametrize("variant", IMPL.keys())
@pytest.mark.parametrize("model_name,config_overrides,seed", [
    ("Llama4 Maverick 17Bx128E", dict(num_local_experts=4), 50),
    ("Llama4 Scout 17Bx16E", dict(num_local_experts=4), 51),
], ids=["maverick", "scout"])
def test_llama4_decoder_from_primitives(variant, model_name, config_overrides, seed):
    """Compose Llama4 decoder (GQA + sigmoid MoE + shared expert, scale_input)."""
    from transformers.models.llama4.configuration_llama4 import Llama4TextConfig
    from transformers.models.llama4.modeling_llama4 import (
        Llama4TextDecoderLayer,
    )

    base_kwargs = dict(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, intermediate_size=64, intermediate_size_mlp=128,
        num_hidden_layers=2, num_local_experts=4, num_experts_per_tok=1,
        moe_layers=[0, 1], layer_types=["full_attention", "full_attention"],
        no_rope_layers=[0, 0],
        use_qk_norm=False, attn_temperature_tuning=False,
        rms_norm_eps=1e-6, max_position_embeddings=256,
        rope_theta=500000.0, router_jitter_noise=0.0,
        attn_implementation="eager",
    )
    if config_overrides:
        base_kwargs.update(config_overrides)

    config = Llama4TextConfig(**base_kwargs)
    seed_all(seed)
    b, seq_len = 1, 4
    x = torch.randn(b, seq_len, config.hidden_size)

    layer = Llama4TextDecoderLayer(config, layer_idx=0)
    with torch.no_grad():
        layer.feed_forward.experts.gate_up_proj.data.normal_()
        layer.feed_forward.experts.down_proj.data.normal_()
    layer.eval()

    # 1) HF reference
    hf_out = layer(x, position_embeddings=None)

    # 2) Collect params
    attn = layer.self_attn
    moe_mod = layer.feed_forward
    mi = config.intermediate_size
    shared = moe_mod.shared_expert
    params = dict(
        eps=config.rms_norm_eps,
        ln1_weight=layer.input_layernorm.weight.data,
        ln2_weight=layer.post_attention_layernorm.weight.data,
        gqa_kwargs=dict(
            wq=attn.q_proj.weight.data, wk=attn.k_proj.weight.data,
            wv=attn.v_proj.weight.data, wo=attn.o_proj.weight.data,
            num_q_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
        ),
        moe_router=moe_mod.router.weight.data,
        moe_w1=moe_mod.experts.gate_up_proj.data[:, :, :mi].permute(0, 2, 1),
        moe_w3=moe_mod.experts.gate_up_proj.data[:, :, mi:].permute(0, 2, 1),
        moe_w2=moe_mod.experts.down_proj.data.permute(0, 2, 1),
        top_k=config.num_experts_per_tok,
        shared_gate=shared.gate_proj.weight.data,
        shared_up=shared.up_proj.weight.data,
        shared_down=shared.down_proj.weight.data,
    )

    # 3) Run composed forward
    h = _compose_llama4_forward(x, params, variant)

    # 4) Compare
    label = variant
    assert_close(hf_out, h, atol=1e-4, rtol=1e-4), \
        f"{model_name} ({label}): composed vs real HF Llama4 mismatch"


# ---------------------------------------------------------------------------
# MiniMax M25 — GQA + MoE (weighted residual, full_attention mode)
# ---------------------------------------------------------------------------

@torch.no_grad()
@pytest.mark.parametrize("variant", IMPL.keys())
def test_minimax_m25_decoder_from_primitives(variant):
    """Compose MiniMax M25 decoder (GQA + MoE with weighted residual)."""
    from transformers import MiniMaxConfig
    from transformers.models.minimax.modeling_minimax import (
        MiniMaxDecoderLayer,
        MiniMaxRotaryEmbedding,
    )

    config = MiniMaxConfig(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        intermediate_size=128, num_local_experts=4, num_experts_per_tok=2,
        rms_norm_eps=1e-6, max_position_embeddings=256, rope_theta=10000.0,
        num_hidden_layers=2, router_jitter_noise=0.0,
        layer_types=["full_attention", "full_attention"],
        full_attn_alpha_factor=1, full_attn_beta_factor=1,
        mlp_alpha_factor=1, mlp_beta_factor=1,
        attn_implementation="eager",
    )

    seed_all(60)
    b, seq_len = 1, 4
    x = torch.randn(b, seq_len, config.hidden_size)

    layer = MiniMaxDecoderLayer(config, layer_idx=0)
    layer.eval()

    rotary = MiniMaxRotaryEmbedding(config)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary(x, position_ids)

    # 1) HF reference
    hf_out = layer(x, position_embeddings=(cos, sin))

    # 2) Collect params
    attn = layer.self_attn
    moe_mod = layer.block_sparse_moe
    w1, w2, w3 = _extract_mixtral_expert_weights(moe_mod.experts, config.num_local_experts)
    params = dict(
        eps=config.rms_norm_eps,
        ln1_weight=layer.input_layernorm.weight.data,
        ln2_weight=layer.post_attention_layernorm.weight.data,
        gqa_kwargs=dict(
            wq=attn.q_proj.weight.data, wk=attn.k_proj.weight.data,
            wv=attn.v_proj.weight.data, wo=attn.o_proj.weight.data,
            num_q_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            position_embeddings=(cos, sin),
        ),
        moe_router=moe_mod.gate.weight.data,
        moe_w1=w1, moe_w2=w2, moe_w3=w3,
        top_k=config.num_experts_per_tok,
        alpha_attn=layer.attn_alpha_factor,
        beta_attn=layer.attn_beta_factor,
        alpha_mlp=layer.mlp_alpha_factor,
        beta_mlp=layer.mlp_beta_factor,
    )

    # 3) Run composed forward
    h = _compose_minimax_forward(x, params, variant)

    # 4) Compare
    label = variant
    assert_close(hf_out, h, atol=1e-4, rtol=1e-4), \
        f"MiniMax M25 ({label}): composed vs real HF MiniMax mismatch"


# ---------------------------------------------------------------------------
# OLMoE 1B-7B — GQA (QK norm→Identity) + MoE (softmax routing)
# ---------------------------------------------------------------------------

@torch.no_grad()
@pytest.mark.parametrize("variant", IMPL.keys())
def test_olmoe_decoder_from_primitives(variant):
    """Compose OLMoE decoder (GQA with QK norm→Identity + softmax MoE)."""
    import torch.nn as nn
    from transformers import OlmoeConfig
    from transformers.models.olmoe.modeling_olmoe import (
        OlmoeDecoderLayer,
        OlmoeRotaryEmbedding,
    )

    config = OlmoeConfig(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, intermediate_size=128, num_experts=4,
        num_experts_per_tok=2, norm_topk_prob=True,
        attention_bias=False, rms_norm_eps=1e-6,
        max_position_embeddings=256, rope_theta=10000.0,
        num_hidden_layers=2, attn_implementation="eager",
    )

    seed_all(65)
    b, seq_len = 1, 4
    x = torch.randn(b, seq_len, config.hidden_size)

    layer = OlmoeDecoderLayer(config, layer_idx=0)
    layer.eval()
    layer.self_attn.q_norm = nn.Identity()
    layer.self_attn.k_norm = nn.Identity()

    rotary = OlmoeRotaryEmbedding(config)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary(x, position_ids)

    # 1) HF reference
    hf_out = layer(x, position_embeddings=(cos, sin))[0]

    # 2) Collect params
    attn = layer.self_attn
    moe_mod = layer.mlp
    w1, w2, w3 = _extract_swiglu_expert_weights(moe_mod.experts, config.num_experts)
    params = dict(
        eps=config.rms_norm_eps,
        ln1_weight=layer.input_layernorm.weight.data,
        ln2_weight=layer.post_attention_layernorm.weight.data,
        gqa_kwargs=dict(
            wq=attn.q_proj.weight.data, wk=attn.k_proj.weight.data,
            wv=attn.v_proj.weight.data, wo=attn.o_proj.weight.data,
            num_q_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            position_embeddings=(cos, sin),
        ),
        moe_router=moe_mod.gate.weight.data,
        moe_w1=w1, moe_w2=w2, moe_w3=w3,
        top_k=config.num_experts_per_tok,
        normalize_routing=config.norm_topk_prob,
    )

    # 3) Run composed forward
    h = _compose_gqa_moe_forward(x, params, variant)

    # 4) Compare
    label = variant
    assert_close(hf_out, h, atol=1e-4, rtol=1e-4), \
        f"OLMoE ({label}): composed vs real HF OLMoE mismatch"


# ---------------------------------------------------------------------------
# Qwen2.5-0.5B — GQA (attention bias→zeroed) + dense SwiGLU
# ---------------------------------------------------------------------------

@torch.no_grad()
@pytest.mark.parametrize("variant", IMPL.keys())
def test_qwen2_5_decoder_from_primitives(variant):
    """
    Compose Qwen2.5 decoder from primitives.

    Adaptations:
    - Qwen2 has hardcoded attention bias=True. Zero out all biases so
      F.linear(x, w, 0) == F.linear(x, w), matching our bias-free gqa functions.
    """
    from transformers import Qwen2Config
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2DecoderLayer,
        Qwen2RotaryEmbedding,
    )

    config = Qwen2Config(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, intermediate_size=128,
        rms_norm_eps=1e-6, max_position_embeddings=256,
        rope_theta=10000.0, num_hidden_layers=2,
        attn_implementation="eager",
    )

    seed_all(70)
    b, seq_len = 1, 4
    x = torch.randn(b, seq_len, config.hidden_size)

    layer = Qwen2DecoderLayer(config, layer_idx=0)
    layer.eval()

    # Zero out attention biases so they don't affect comparison
    layer.self_attn.q_proj.bias.data.zero_()
    layer.self_attn.k_proj.bias.data.zero_()
    layer.self_attn.v_proj.bias.data.zero_()

    rotary = Qwen2RotaryEmbedding(config)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary(x, position_ids)

    # 1) HF reference
    hf_out = layer(x, position_embeddings=(cos, sin))

    # 2) Collect params
    params = _collect_gqa_dense_params(layer, config, position_embeddings=(cos, sin))

    # 3) Run composed forward
    h = _compose_gqa_dense_forward(x, params, variant)

    # 4) Compare
    label = variant
    assert_close(hf_out, h, atol=1e-4, rtol=1e-4), \
        f"Qwen2.5 ({label}): composed vs real HF Qwen2 mismatch"


# ---------------------------------------------------------------------------
# Qwen3-235B-A22B / Qwen3-Coder-480B-A35B / Qwen3-Next-80B-A3B
#   — GQA (QK norm→Identity) + MoE (softmax routing)
# ---------------------------------------------------------------------------

@torch.no_grad()
@pytest.mark.parametrize("variant", IMPL.keys())
@pytest.mark.parametrize("model_name,config_overrides,seed", [
    ("Qwen3-235B-A22B", None, 80),
    ("Qwen3-Coder-480B-A35B", None, 81),
    ("Qwen3-Next-80B-A3B", dict(num_experts=8, num_experts_per_tok=2), 82),
], ids=["qwen3_235b", "qwen3_coder_480b", "qwen3_next_80b"])
def test_qwen3moe_decoder_from_primitives(variant, model_name, config_overrides, seed):
    """Compose Qwen3Moe decoder (QK norm→Identity + softmax MoE)."""
    import torch.nn as nn
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeDecoderLayer,
        Qwen3MoeRotaryEmbedding,
    )

    base_kwargs = dict(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, intermediate_size=128, moe_intermediate_size=64,
        num_experts=4, num_experts_per_tok=2, norm_topk_prob=False,
        attention_bias=False, rms_norm_eps=1e-6,
        max_position_embeddings=256, rope_theta=10000.0,
        num_hidden_layers=2, attn_implementation="eager",
    )
    if config_overrides:
        base_kwargs.update(config_overrides)

    config = Qwen3MoeConfig(**base_kwargs)
    seed_all(seed)
    b, seq_len = 1, 4
    x = torch.randn(b, seq_len, config.hidden_size)

    layer = Qwen3MoeDecoderLayer(config, layer_idx=0)
    layer.eval()
    layer.self_attn.q_norm = nn.Identity()
    layer.self_attn.k_norm = nn.Identity()

    rotary = Qwen3MoeRotaryEmbedding(config)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary(x, position_ids)

    # 1) HF reference
    hf_out = layer(x, position_embeddings=(cos, sin))

    # 2) Collect params
    attn = layer.self_attn
    moe_mod = layer.mlp
    w1, w2, w3 = _extract_swiglu_expert_weights(moe_mod.experts, config.num_experts)
    params = dict(
        eps=config.rms_norm_eps,
        ln1_weight=layer.input_layernorm.weight.data,
        ln2_weight=layer.post_attention_layernorm.weight.data,
        gqa_kwargs=dict(
            wq=attn.q_proj.weight.data, wk=attn.k_proj.weight.data,
            wv=attn.v_proj.weight.data, wo=attn.o_proj.weight.data,
            num_q_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            position_embeddings=(cos, sin),
        ),
        moe_router=moe_mod.gate.weight.data,
        moe_w1=w1, moe_w2=w2, moe_w3=w3,
        top_k=config.num_experts_per_tok,
        normalize_routing=config.norm_topk_prob,
    )

    # 3) Run composed forward
    h = _compose_gqa_moe_forward(x, params, variant)

    # 4) Compare
    label = variant
    assert_close(hf_out, h, atol=1e-4, rtol=1e-4), \
        f"{model_name} ({label}): composed vs real HF Qwen3Moe mismatch"


# ---------------------------------------------------------------------------
# Qwen3-32B — GQA (QK norm→Identity) + dense SwiGLU
# ---------------------------------------------------------------------------

@torch.no_grad()
@pytest.mark.parametrize("variant", IMPL.keys())
def test_qwen3_32b_decoder_from_primitives(variant):
    """
    Compose Qwen3-32B decoder from primitives.

    Adaptations:
    - QK norm monkey-patched to nn.Identity
    """
    import torch.nn as nn
    from transformers import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3DecoderLayer,
        Qwen3RotaryEmbedding,
    )

    config = Qwen3Config(
        hidden_size=64, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, intermediate_size=128, attention_bias=False,
        rms_norm_eps=1e-6, max_position_embeddings=256,
        rope_theta=10000.0, num_hidden_layers=2,
        attn_implementation="eager",
    )

    seed_all(85)
    b, seq_len = 1, 4
    x = torch.randn(b, seq_len, config.hidden_size)

    layer = Qwen3DecoderLayer(config, layer_idx=0)
    layer.eval()

    layer.self_attn.q_norm = nn.Identity()
    layer.self_attn.k_norm = nn.Identity()

    rotary = Qwen3RotaryEmbedding(config)
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary(x, position_ids)

    # 1) HF reference
    hf_out = layer(x, position_embeddings=(cos, sin))

    # 2) Collect params
    params = _collect_gqa_dense_params(layer, config, position_embeddings=(cos, sin))

    # 3) Run composed forward
    h = _compose_gqa_dense_forward(x, params, variant)

    # 4) Compare
    label = variant
    assert_close(hf_out, h, atol=1e-4, rtol=1e-4), \
        f"Qwen3-32B ({label}): composed vs real HF Qwen3 mismatch"
