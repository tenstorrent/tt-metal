# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pure-torch equivalence test for the CPU reference (reference/model.py).

Reproduces the validated golden math from ``test_model_vs_ref.py`` (the inline torch reference
that the TT ``Model`` is PCC-checked against) on its exact reduced config + random weights, then
asserts ``MiniMaxM3TextModel.forward_hidden`` produces identical logits and per-layer K/V. This
pins the reference (used to generate golden KV caches) to the same math the device model targets.

No device / ttnn needed — runs anywhere torch is available.
"""

import torch

from models.demos.minimax_m3.reference.model import DictWeights, MiniMaxM3Config, MiniMaxM3TextModel

# Reduced config — copied verbatim from tests/unit/test_model_vs_ref.py (the validated golden).
HIDDEN, NQ, NKV, HEAD_DIM, ROTARY_DIM, THETA, EPS = 6144, 64, 4, 128, 64, 5_000_000.0, 1e-6
INTER, SHARED_INTER, E, TOPK, SCALE, ALPHA, LIMIT, V = 512, 512, 8, 2, 2.0, 1.702, 7.0, 2048
SCHEDULE = [0, 1]  # 0 -> dense layer, 1 -> MoE layer


def _gemma_norm(x, w):
    var = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + EPS)) * (1.0 + w.float())


def _gemma_head_norm(x, w):
    var = x.pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(var + EPS)) * (1.0 + w.float())


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _rope(t, cos, sin):
    r, p = t[..., :ROTARY_DIM], t[..., ROTARY_DIM:]
    return torch.cat([r * cos + _rotate_half(r) * sin, p], dim=-1)


def _attn(x, w, cos, sin):
    B, S, _ = x.shape
    q = (x @ w["q"].t()).view(B, S, NQ, HEAD_DIM).transpose(1, 2)
    k = (x @ w["k"].t()).view(B, S, NKV, HEAD_DIM).transpose(1, 2)
    v = (x @ w["v"].t()).view(B, S, NKV, HEAD_DIM).transpose(1, 2)
    q = _rope(_gemma_head_norm(q, w["q_norm"]), cos, sin)
    k = _rope(_gemma_head_norm(k, w["k_norm"]), cos, sin)
    rep = NQ // NKV
    k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
    s = (q @ k.transpose(-1, -2)) * (HEAD_DIM**-0.5)
    s = s + torch.triu(torch.full((S, S), float("-inf")), diagonal=1)
    o = (torch.softmax(s, dim=-1) @ v).transpose(1, 2).reshape(B, S, NQ * HEAD_DIM)
    return o @ w["o"].t()


def _swiglu(g, u):
    g = g.clamp(max=LIMIT)
    u = u.clamp(min=-LIMIT, max=LIMIT)
    return (u + 1.0) * (g * torch.sigmoid(ALPHA * g))


def _ffn(x, w1, w3, w2):
    return _swiglu(x @ w1.t(), x @ w3.t()) @ w2.t()


def _moe(x, w):
    scores = torch.sigmoid(x @ w["gate"].t())
    _, idx = torch.topk(scores + w["bias"], TOPK, dim=-1)
    tw = torch.gather(scores, 1, idx)
    tw = (tw / tw.sum(-1, keepdim=True)) * SCALE
    routed = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for j in range(TOPK):
            routed[t] += tw[t, j] * _ffn(x[t : t + 1], *w["experts"][idx[t, j].item()]).squeeze(0)
    return routed + _ffn(x, *w["shared"])


def _layer(x, w, cos, sin, is_dense):
    x = x + _attn(_gemma_norm(x, w["input_ln"]), w, cos, sin)
    normed = _gemma_norm(x, w["post_ln"])
    ffn = _ffn(normed.reshape(-1, HIDDEN), *w["dense"]) if is_dense else _moe(normed.reshape(-1, HIDDEN), w)
    return x + ffn.reshape(x.shape)


def _rand(*s):
    return torch.randn(*s) * 0.02


def _attn_weights():
    return {
        "input_ln": torch.randn(HIDDEN) * 0.1,
        "post_ln": torch.randn(HIDDEN) * 0.1,
        "q": _rand(NQ * HEAD_DIM, HIDDEN),
        "k": _rand(NKV * HEAD_DIM, HIDDEN),
        "v": _rand(NKV * HEAD_DIM, HIDDEN),
        "o": _rand(HIDDEN, NQ * HEAD_DIM),
        "q_norm": torch.randn(HEAD_DIM) * 0.1,
        "k_norm": torch.randn(HEAD_DIM) * 0.1,
    }


def _rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


def test_reference_matches_validated_golden():
    """forward_hidden == the validated inline torch reference (logits + per-layer K/V), fp32."""
    torch.manual_seed(0)
    seq_len = 128
    x = torch.randn(1, seq_len, HIDDEN) * 0.1

    layer_w = []
    for val in SCHEDULE:
        w = _attn_weights()
        if val == 0:
            w["dense"] = (_rand(INTER, HIDDEN), _rand(INTER, HIDDEN), _rand(HIDDEN, INTER))
        else:
            w["gate"] = torch.randn(E, HIDDEN) * 0.05
            w["bias"] = torch.randn(E) * 0.1
            w["experts"] = [(_rand(INTER, HIDDEN), _rand(INTER, HIDDEN), _rand(HIDDEN, INTER)) for _ in range(E)]
            w["shared"] = (_rand(SHARED_INTER, HIDDEN), _rand(SHARED_INTER, HIDDEN), _rand(HIDDEN, SHARED_INTER))
        layer_w.append(w)
    final_norm_w = torch.randn(HIDDEN) * 0.1
    lm_head_w = _rand(V, HIDDEN)

    # --- validated inline reference forward, capturing K/V the same way the model does ---
    inv_freq = 1.0 / (THETA ** (torch.arange(0, ROTARY_DIM, 2).float() / ROTARY_DIM))
    emb = torch.cat([torch.outer(torch.arange(seq_len).float(), inv_freq)] * 2, dim=-1)
    cos_ref, sin_ref = emb.cos()[None, None], emb.sin()[None, None]
    h = x.float()
    ref_kv = []
    for w, val in zip(layer_w, SCHEDULE):
        B, S, _ = h.shape
        nn = _gemma_norm(h, w["input_ln"])
        k = _rope(
            _gemma_head_norm((nn @ w["k"].t()).view(B, S, NKV, HEAD_DIM).transpose(1, 2), w["k_norm"]), cos_ref, sin_ref
        )
        v = (nn @ w["v"].t()).view(B, S, NKV, HEAD_DIM).transpose(1, 2)
        ref_kv.append((k, v))
        h = _layer(h, w, cos_ref, sin_ref, val == 0)
    ref_logits = _gemma_norm(h, final_norm_w) @ lm_head_w.t()

    # --- our reference from the same raw-HF weights ---
    state = {
        "model.embed_tokens.weight": _rand(V, HIDDEN),
        "model.norm.weight": final_norm_w,
        "lm_head.weight": lm_head_w,
    }
    for i, (w, val) in enumerate(zip(layer_w, SCHEDULE)):
        p = f"model.layers.{i}."
        state[p + "input_layernorm.weight"] = w["input_ln"]
        state[p + "post_attention_layernorm.weight"] = w["post_ln"]
        state[p + "self_attn.q_proj.weight"] = w["q"]
        state[p + "self_attn.k_proj.weight"] = w["k"]
        state[p + "self_attn.v_proj.weight"] = w["v"]
        state[p + "self_attn.o_proj.weight"] = w["o"]
        state[p + "self_attn.q_norm.weight"] = w["q_norm"]
        state[p + "self_attn.k_norm.weight"] = w["k_norm"]
        if val == 0:
            g, u, d = w["dense"]
            state[p + "mlp.gate_proj.weight"] = g
            state[p + "mlp.up_proj.weight"] = u
            state[p + "mlp.down_proj.weight"] = d
        else:
            state[p + "block_sparse_moe.gate.weight"] = w["gate"]
            state[p + "block_sparse_moe.e_score_correction_bias"] = w["bias"]
            state[p + "block_sparse_moe.shared_experts.gate_proj.weight"] = w["shared"][0]
            state[p + "block_sparse_moe.shared_experts.up_proj.weight"] = w["shared"][1]
            state[p + "block_sparse_moe.shared_experts.down_proj.weight"] = w["shared"][2]
            for e, (w1, w3, w2) in enumerate(w["experts"]):
                state[p + f"block_sparse_moe.experts.{e}.w1.weight"] = w1
                state[p + f"block_sparse_moe.experts.{e}.w3.weight"] = w3
                state[p + f"block_sparse_moe.experts.{e}.w2.weight"] = w2

    cfg = MiniMaxM3Config(
        hidden_size=HIDDEN,
        num_hidden_layers=len(SCHEDULE),
        num_attention_heads=NQ,
        num_key_value_heads=NKV,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        rope_theta=THETA,
        rms_norm_eps=EPS,
        vocab_size=V,
        num_local_experts=E,
        num_experts_per_tok=TOPK,
        routed_scaling_factor=SCALE,
        swiglu_alpha=ALPHA,
        swiglu_limit=LIMIT,
        moe_layer_freq=list(SCHEDULE),
        sparse_attention_freq=[0, 0],
    )
    model = MiniMaxM3TextModel(cfg, DictWeights(state, dtype=torch.float32))
    kv, logits = model.forward_hidden(x.float(), compute_logits=True)

    assert _rel_err(logits, ref_logits) < 1e-5
    for (k, v), (rk, rv) in zip(kv, ref_kv):
        assert _rel_err(k, rk) < 1e-5
        assert _rel_err(v, rv) < 1e-5


def _sparse_layer_model(block_size, topk_blocks, *, hidden=64, nq=8, nkv=2, hd=8, rotary=4, idx=8, inter=32):
    """A 1-layer model with an MSA sparse-attention layer (+ trivial dense MLP), random weights."""
    cfg = MiniMaxM3Config(
        hidden_size=hidden,
        num_hidden_layers=1,
        num_attention_heads=nq,
        num_key_value_heads=nkv,
        head_dim=hd,
        rotary_dim=rotary,
        vocab_size=4,
        moe_layer_freq=[0],
        sparse_attention_freq=[1],
        sparse_index_dim=idx,
        sparse_num_index_heads=nkv,
        sparse_block_size=block_size,
        sparse_topk_blocks=topk_blocks,
        sparse_local_block=1,
    )
    p = "model.layers.0."
    state = {
        p + "input_layernorm.weight": torch.randn(hidden) * 0.1,
        p + "post_attention_layernorm.weight": torch.randn(hidden) * 0.1,
        p + "self_attn.q_proj.weight": torch.randn(nq * hd, hidden) * 0.05,
        p + "self_attn.k_proj.weight": torch.randn(nkv * hd, hidden) * 0.05,
        p + "self_attn.v_proj.weight": torch.randn(nkv * hd, hidden) * 0.05,
        p + "self_attn.o_proj.weight": torch.randn(hidden, nq * hd) * 0.05,
        p + "self_attn.q_norm.weight": torch.randn(hd) * 0.1,
        p + "self_attn.k_norm.weight": torch.randn(hd) * 0.1,
        p + "self_attn.index_q_proj.weight": torch.randn(nkv * idx, hidden) * 0.05,
        p + "self_attn.index_k_proj.weight": torch.randn(idx, hidden) * 0.05,
        p + "self_attn.index_q_norm.weight": torch.randn(idx) * 0.1,
        p + "self_attn.index_k_norm.weight": torch.randn(idx) * 0.1,
        p + "mlp.gate_proj.weight": torch.randn(inter, hidden) * 0.05,
        p + "mlp.up_proj.weight": torch.randn(inter, hidden) * 0.05,
        p + "mlp.down_proj.weight": torch.randn(hidden, inter) * 0.05,
        "model.norm.weight": torch.randn(hidden) * 0.1,
        "lm_head.weight": torch.randn(cfg.vocab_size, hidden) * 0.05,
    }
    return MiniMaxM3TextModel(cfg, DictWeights(state, dtype=torch.float32)), cfg


def test_msa_reduces_to_dense_below_bound():
    """At seq_len <= topk_blocks*block_size, MSA selects every block -> identical to dense_only."""
    torch.manual_seed(0)
    model, cfg = _sparse_layer_model(block_size=8, topk_blocks=16)  # bound = 128
    S = 64  # <= 128 => every block selected
    assert S <= cfg.sparse_topk_blocks * cfg.sparse_block_size
    x = torch.randn(1, S, cfg.hidden_size)
    (kv_msa, _), (kv_dense, _) = model.forward_hidden(x), model.forward_hidden(x, dense_only=True)
    # KV cache is identical (post-RoPE K + raw V don't depend on the attention mask)...
    assert _rel_err(kv_msa[0][0], kv_dense[0][0]) < 1e-6 and _rel_err(kv_msa[0][1], kv_dense[0][1]) < 1e-6
    # ...and so are the hidden states / logits, because all blocks are selected (MSA == dense causal).
    h_msa = model.forward_hidden(x, compute_logits=True)[1]
    h_dense = model.forward_hidden(x, dense_only=True, compute_logits=True)[1]
    assert _rel_err(h_msa, h_dense) < 1e-6


def test_msa_is_sparse_above_bound():
    """Above the bound MSA drops blocks, so it must differ from dense_only (and stay finite/causal)."""
    torch.manual_seed(0)
    model, cfg = _sparse_layer_model(block_size=8, topk_blocks=2)  # bound = 16
    S = 64  # > 16 => genuine sparsity (only top-2 + local blocks attended)
    assert S > cfg.sparse_topk_blocks * cfg.sparse_block_size
    x = torch.randn(1, S, cfg.hidden_size)
    h_msa = model.forward_hidden(x, compute_logits=True)[1]
    h_dense = model.forward_hidden(x, dense_only=True, compute_logits=True)[1]
    assert torch.isfinite(h_msa).all()
    assert _rel_err(h_msa, h_dense) > 1e-3, "MSA should diverge from dense attention above the bound"
