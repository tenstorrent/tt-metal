# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pure-torch MiMo-V2.6-Flash text reference.

Mirrors ``modeling_mimo_v2.py`` from the HF repo (eager attention path) without the transformers
dependency: RMSNorm, fused-QKV attention with partial RoPE (first 64 of 192 dims, ``rotate_half``),
``attention_value_scale`` on V, sliding-window (128) attention with a learned sink logit on SWA
layers, a dense SwiGLU MLP on layer 0 and sigmoid ``noaux_tc`` top-8 routing over 256 experts.
Everything runs in fp32 on dequantized weights. Each user keeps its own KV history.
"""

import torch
import torch.nn.functional as F

# Attention runs this many queries at a time. Scores for all queries at once are [64, S, T] fp32,
# 275 GB at S = T = 32K, plus a full-size copy per mask/softmax step. One block of 1024 queries
# against 32K keys is 8.6 GB.
QUERY_BLOCK = 1024


def rms_norm(x, weight, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return weight * xf.to(x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin(positions, rope_dim, theta):
    """positions [S] -> cos, sin [S, rope_dim] (HF layout: cat(freqs, freqs))."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.int64).float() / rope_dim))
    freqs = positions.float()[:, None] * inv_freq[None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def apply_partial_rope(x, cos, sin, rope_dim):
    """x [heads, S, head_dim]; rotate the first rope_dim dims."""
    x_rope, x_nope = x[..., :rope_dim], x[..., rope_dim:]
    x_rope = x_rope * cos + rotate_half(x_rope) * sin
    return torch.cat((x_rope, x_nope), dim=-1)


class KVHistory:
    """Per-user, per-layer K/V history (post-RoPE K, post-scale V), as HF's DynamicCache stores it."""

    def __init__(self):
        self.k = None  # [kv_heads, T, head_dim]
        self.v = None  # [kv_heads, T, v_head_dim]
        self.pos = None  # [T]

    def append(self, k, v, pos):
        if self.k is None:
            self.k, self.v, self.pos = k, v, pos
        else:
            self.k = torch.cat((self.k, k), dim=1)
            self.v = torch.cat((self.v, v), dim=1)
            self.pos = torch.cat((self.pos, pos))
        return self.k, self.v, self.pos


def attention(x, w, dims, v_scale, positions, history: KVHistory | None = None):
    """x [S, hidden] for one user at ``positions`` [S]; returns [S, hidden]."""
    S = x.shape[0]
    q = (x @ w["q_proj"].T).view(S, dims.num_heads, dims.head_dim).transpose(0, 1)
    k = (x @ w["k_proj"].T).view(S, dims.num_kv_heads, dims.head_dim).transpose(0, 1)
    v = (x @ w["v_proj"].T).view(S, dims.num_kv_heads, dims.v_head_dim).transpose(0, 1)
    if v_scale is not None:
        v = v * v_scale

    cos, sin = rope_cos_sin(positions, dims.rope_dim, dims.rope_theta)
    q = apply_partial_rope(q, cos, sin, dims.rope_dim)
    k = apply_partial_rope(k, cos, sin, dims.rope_dim)

    if history is not None:
        k_all, v_all, k_pos = history.append(k, v, positions)
    else:
        k_all, v_all, k_pos = k, v, positions

    groups = dims.num_heads // dims.num_kv_heads
    k_rep = k_all.repeat_interleave(groups, dim=0)
    v_rep = v_all.repeat_interleave(groups, dim=0)
    sinks = w["attention_sink_bias"].reshape(-1, 1, 1) if dims.has_sink else None

    out = torch.empty(dims.num_heads, S, dims.v_head_dim, dtype=torch.float32)
    for q0 in range(0, S, QUERY_BLOCK):
        q1 = min(q0 + QUERY_BLOCK, S)
        out[:, q0:q1] = _attend(q[:, q0:q1], positions[q0:q1], k_rep, v_rep, k_pos, dims, sinks)
    return out.transpose(0, 1).reshape(S, dims.num_heads * dims.v_head_dim) @ w["o_proj"].T


def _attend(q, positions, k_rep, v_rep, k_pos, dims, sinks):
    """Softmax attention for one block of queries: q [heads, s, head_dim] at ``positions`` [s].

    Keys no query in the block can see (after its last position, or before its window on SWA layers)
    are left out. They would get probability exactly 0, so the result is the same."""
    keep = k_pos <= positions.max()
    if dims.sliding_window is not None:
        keep &= k_pos > positions.min() - dims.sliding_window
    k_rep, v_rep, k_pos = k_rep[:, keep], v_rep[:, keep], k_pos[keep]
    scores = (q @ k_rep.transpose(1, 2)) * dims.scaling  # [heads, s, t]

    allowed = k_pos[None, :] <= positions[:, None]
    if dims.sliding_window is not None:
        allowed &= (positions[:, None] - k_pos[None, :]) < dims.sliding_window
    scores = scores.masked_fill(~allowed[None], float("-inf"))

    if sinks is not None:
        scores = torch.cat((scores, sinks.expand(-1, q.shape[1], 1)), dim=-1)
    scores = scores - scores.max(dim=-1, keepdim=True).values
    probs = F.softmax(scores, dim=-1, dtype=torch.float32)
    if sinks is not None:
        probs = probs[..., :-1]
    return probs @ v_rep


def dense_mlp(x, w):
    return (F.silu(x @ w["gate_proj"].T) * (x @ w["up_proj"].T)) @ w["down_proj"].T


def route(x, router, cfg):
    """HF ``MiMoV2MoEGate.forward`` for n_group == topk_group == 1. x [T, hidden] -> (idx [T, k], weight [T, k])."""
    logits = F.linear(x.float(), router["weight"].float())
    scores = logits.sigmoid()
    scores_for_choice = scores + router["e_score_correction_bias"][None, :]
    _, topk_idx = torch.topk(scores_for_choice, k=cfg.num_experts_per_tok, dim=-1, sorted=False)
    topk_weight = scores.gather(1, topk_idx)
    if cfg.num_experts_per_tok > 1 and cfg.norm_topk_prob:
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
    return topk_idx, topk_weight * cfg.routed_scaling_factor


def moe(x, router, experts, cfg):
    """x [T, hidden]; ``experts`` holds stacked gate/up [E, I, H] and down [E, H, I] tensors."""
    topk_idx, topk_weight = route(x, router, cfg)
    out = torch.zeros_like(x, dtype=torch.float32)
    for e in torch.unique(topk_idx).tolist():
        token_idx, slot = torch.where(topk_idx == e)
        xe = x[token_idx].float()
        gate = xe @ experts["gate_proj"][e].float().T
        up = xe @ experts["up_proj"][e].float().T
        ye = (F.silu(gate) * up) @ experts["down_proj"][e].float().T
        out.index_add_(0, token_idx, ye * topk_weight[token_idx, slot, None])
    return out


class ReferenceDecoderLayer:
    def __init__(self, cfg, layer_idx, ckpt=None, weights=None):
        """Load from ``ckpt`` (a MiMoCheckpoint), or pass prepared ``weights`` (same keys) for synthetic tests."""
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.dims = cfg.attention_dims(layer_idx)
        self.is_moe = cfg.is_moe(layer_idx)
        if weights is None:
            weights = load_layer_weights(ckpt, cfg, layer_idx)
        self.w = weights

    def __call__(self, x, positions, history=None):
        cfg = self.cfg
        h = rms_norm(x, self.w["input_layernorm"], cfg.layernorm_epsilon)
        x = x + attention(h, self.w["attn"], self.dims, cfg.attention_value_scale, positions, history)
        h = rms_norm(x, self.w["post_attention_layernorm"], cfg.layernorm_epsilon)
        if self.is_moe:
            x = x + moe(h, self.w["router"], self.w["experts"], cfg)
        else:
            x = x + dense_mlp(h, self.w["mlp"])
        return x


def load_layer_weights(ckpt, cfg, layer_idx, expert_dtype=torch.bfloat16):
    w = dict(ckpt.norms(layer_idx))
    w["attn"] = ckpt.attention(layer_idx)
    if cfg.is_moe(layer_idx):
        w["router"] = ckpt.router(layer_idx)
        w["experts"] = ckpt.experts(layer_idx, dtype=expert_dtype)
    else:
        w["mlp"] = ckpt.dense_mlp(layer_idx)
    return w
