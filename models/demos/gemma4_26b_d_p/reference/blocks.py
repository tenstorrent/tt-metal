# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Functional torch reference for Gemma-4 MoE text model, split into bring-up *blocks*.

Every block is an ``nn.Module`` with explicit tensor inputs (no cache objects, no
mask factories, no data-dependent Python control flow except inside ``Experts``,
which the fx tracer treats as a leaf). Chunked prefill is expressed explicitly:
an attention block receives the KV *prefix* from earlier chunks and returns the
new K/V for its own chunk, so the same code serves full-sequence and chunked runs.

The module tree mirrors HF state-dict names, so ``load_state_dict`` works on the
``model.language_model.`` sub-tree directly (see ``weights.py``).

Block ids (``BLOCK_*``) are the keys used by the bring-up dashboard.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import FULL, SLIDING, Gemma4TextConfig

# ---------------------------------------------------------------------------
# Leaf blocks
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """``x * rsqrt(mean(x^2) + eps) [* w]`` in fp32. No Gemma (1+w) offset."""

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.pow(xf.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if self.with_scale:
            y = y * self.weight.float()
        return y.type_as(x)


class ScaledEmbedding(nn.Module):
    def __init__(self, vocab: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(vocab, dim))
        self.scale = dim**0.5

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        e = F.embedding(ids, self.weight)
        return e * torch.tensor(self.scale, dtype=e.dtype)


def rope_cos_sin(positions: torch.Tensor, theta: float, head_dim: int, rotated_pairs: int):
    """HF-layout cos/sin ``[S, head_dim]`` (``cat(freqs, freqs)``); unrotated pairs get cos=1, sin=0."""
    j = torch.arange(0, 2 * rotated_pairs, 2, dtype=torch.float32)
    inv = 1.0 / (theta ** (j / head_dim))
    inv = torch.cat([inv, torch.zeros(head_dim // 2 - rotated_pairs)])
    freqs = positions.float()[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class Rope(nn.Module):
    """Apply rotate-half RoPE to ``[B, H, S, D]`` given ``cos/sin [S, D]``."""

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        cos = cos.to(x.dtype)[None, None]
        sin = sin.to(x.dtype)[None, None]
        return x * cos + rotate_half(x) * sin


def attention_mask(q_pos: torch.Tensor, k_pos: torch.Tensor, window: int | None) -> torch.Tensor:
    """Additive mask ``[Sq, Sk]``: causal, optionally limited to ``q - k < window``."""
    d = q_pos[:, None] - k_pos[None, :]
    ok = d >= 0
    if window is not None:
        ok = ok & (d < window)
    return torch.where(ok, 0.0, float("-inf"))


class SDPA(nn.Module):
    """GQA attention with scaling 1.0 (Gemma-4 normalises Q/K instead of 1/sqrt(d))."""

    def __init__(self, n_q: int, n_kv: int, scaling: float = 1.0):
        super().__init__()
        self.groups = n_q // n_kv
        self.scaling = scaling

    def forward(self, q, k, v, mask):
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)
        s = torch.matmul(q.float(), k.float().transpose(-1, -2)) * self.scaling + mask
        p = torch.softmax(s, dim=-1)
        return torch.matmul(p, v.float()).to(q.dtype)


class Attention(nn.Module):
    """One attention sub-block for a chunk.

    Inputs: ``x [B,S,H]`` (already input-normed), ``cos/sin [S,D]``, ``k_prefix/v_prefix
    [B,Hkv,P,D]`` (post-norm/rope cache of earlier chunks; P may be 0), ``mask [S, P+S]``.
    Returns ``(out [B,S,H], k_chunk, v_chunk)`` where ``k_chunk/v_chunk`` are what gets
    written to the KV cache for this chunk.
    """

    def __init__(self, cfg: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = cfg.layer_types[layer_idx]
        self.n_q = cfg.num_attention_heads
        self.n_kv = cfg.layer_kv_heads(layer_idx)
        self.hd = cfg.layer_head_dim(layer_idx)
        self.k_eq_v = cfg.layer_k_eq_v(layer_idx)
        H = cfg.hidden_size
        self.q_proj = nn.Linear(H, self.n_q * self.hd, bias=False)
        self.k_proj = nn.Linear(H, self.n_kv * self.hd, bias=False)
        self.v_proj = None if self.k_eq_v else nn.Linear(H, self.n_kv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.n_q * self.hd, H, bias=False)
        self.q_norm = RMSNorm(self.hd, cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.hd, cfg.rms_norm_eps)
        self.v_norm = RMSNorm(self.hd, cfg.rms_norm_eps, with_scale=False)
        self.rope = Rope()
        self.sdpa = SDPA(self.n_q, self.n_kv)

    def forward(self, x, cos, sin, k_prefix, v_prefix, mask):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_q, self.hd).transpose(1, 2)
        k_raw = self.k_proj(x).view(B, S, self.n_kv, self.hd).transpose(1, 2)
        v_raw = k_raw if self.v_proj is None else self.v_proj(x).view(B, S, self.n_kv, self.hd).transpose(1, 2)
        q = self.rope(self.q_norm(q), cos, sin)
        k = self.rope(self.k_norm(k_raw), cos, sin)
        v = self.v_norm(v_raw)
        k_all = torch.cat([k_prefix, k], dim=2)
        v_all = torch.cat([v_prefix, v], dim=2)
        o = self.sdpa(q, k_all, v_all, mask)
        o = o.transpose(1, 2).reshape(B, S, self.n_q * self.hd)
        return self.o_proj(o), k, v


class DenseMLP(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class Router(nn.Module):
    """Returns dense routing weights ``[T, E]`` (zeros off the top-k) and top-k indices ``[T, k]``."""

    def __init__(self, cfg: Gemma4TextConfig):
        super().__init__()
        H, E = cfg.hidden_size, cfg.num_experts
        self.top_k = cfg.top_k_experts
        self.norm = RMSNorm(H, cfg.rms_norm_eps, with_scale=False)
        self.proj = nn.Linear(H, E, bias=False)
        self.scale = nn.Parameter(torch.ones(H))
        self.per_expert_scale = nn.Parameter(torch.ones(E))
        self.scalar_root_size = H**-0.5

    def forward(self, x):
        h = self.norm(x) * self.scale * self.scalar_root_size
        probs = torch.softmax(self.proj(h).float(), dim=-1)
        w, idx = torch.topk(probs, self.top_k, dim=-1)
        w = w / w.sum(dim=-1, keepdim=True)
        w = w * self.per_expert_scale[idx]
        dense = torch.zeros_like(probs).scatter(-1, idx, w)
        return dense.to(x.dtype), idx


class Experts(nn.Module):
    """128 gated-GELU experts. ``gate_up_proj [E, 2I, H]`` (gate rows first), ``down_proj [E, H, I]``.

    fx leaf: the token->expert grouping is data dependent.
    """

    def __init__(self, cfg: Gemma4TextConfig):
        super().__init__()
        E, H, I = cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.zeros(E, 2 * I, H))
        self.down_proj = nn.Parameter(torch.zeros(E, H, I))

    def forward(self, x, routing, idx):
        T, H = x.shape
        out = torch.zeros(T, H, dtype=torch.float32)
        for e in torch.unique(idx).tolist():
            tok = (idx == e).any(-1).nonzero(as_tuple=True)[0]
            xe = x[tok]
            gate, up = F.linear(xe, self.gate_up_proj[e]).chunk(2, dim=-1)
            ye = F.linear(F.gelu(gate, approximate="tanh") * up, self.down_proj[e])
            out.index_add_(0, tok, ye.float() * routing[tok, e : e + 1].float())
        return out.to(x.dtype)


# ---------------------------------------------------------------------------
# Composite blocks
# ---------------------------------------------------------------------------


class MoEBranch(nn.Module):
    """router(residual) -> pre_ff_ln_2 -> experts -> post_ff_ln_2 (norms owned by the layer)."""

    def forward(self, x_normed, routing, idx, experts: Experts):
        B, S, H = x_normed.shape
        return experts(x_normed.reshape(B * S, H), routing.reshape(B * S, -1), idx.reshape(B * S, -1)).view(B, S, H)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: Gemma4TextConfig, layer_idx: int):
        super().__init__()
        H, eps = cfg.hidden_size, cfg.rms_norm_eps
        self.layer_idx = layer_idx
        self.layer_type = cfg.layer_types[layer_idx]
        self.input_layernorm = RMSNorm(H, eps)
        self.self_attn = Attention(cfg, layer_idx)
        self.post_attention_layernorm = RMSNorm(H, eps)
        self.pre_feedforward_layernorm = RMSNorm(H, eps)
        self.mlp = DenseMLP(H, cfg.intermediate_size)
        self.post_feedforward_layernorm_1 = RMSNorm(H, eps)
        self.router = Router(cfg)
        self.pre_feedforward_layernorm_2 = RMSNorm(H, eps)
        self.experts = Experts(cfg)
        self.post_feedforward_layernorm_2 = RMSNorm(H, eps)
        self.post_feedforward_layernorm = RMSNorm(H, eps)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, x, cos, sin, k_prefix, v_prefix, mask):
        B, S, H = x.shape
        a, k, v = self.self_attn(self.input_layernorm(x), cos, sin, k_prefix, v_prefix, mask)
        r = x + self.post_attention_layernorm(a)
        m1 = self.post_feedforward_layernorm_1(self.mlp(self.pre_feedforward_layernorm(r)))
        routing, idx = self.router(r)
        xe = self.pre_feedforward_layernorm_2(r).reshape(B * S, H)
        m2 = self.experts(xe, routing.reshape(B * S, -1), idx.reshape(B * S, -1)).view(B, S, H)
        m2 = self.post_feedforward_layernorm_2(m2)
        out = r + self.post_feedforward_layernorm(m1 + m2)
        return out * self.layer_scalar.to(out.dtype), k, v


class LMHead(nn.Module):
    def __init__(self, cfg: Gemma4TextConfig, embed_weight: nn.Parameter | None = None):
        super().__init__()
        self.weight = embed_weight if embed_weight is not None else nn.Parameter(torch.zeros(cfg.vocab_size, cfg.hidden_size))
        self.softcap = cfg.final_logit_softcapping

    def forward(self, x):
        logits = F.linear(x, self.weight)
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits


# ---------------------------------------------------------------------------
# Model + chunked prefill driver
# ---------------------------------------------------------------------------


class Gemma4TextModel(nn.Module):
    """Mirrors ``model.language_model`` of the HF checkpoint (state-dict compatible)."""

    def __init__(self, cfg: Gemma4TextConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = ScaledEmbedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = LMHead(cfg, self.embed_tokens.weight if cfg.tie_word_embeddings else None)
        self.rope_specs = {t: cfg.rope_spec(t) for t in (SLIDING, FULL)}

    def empty_kv(self, batch: int = 1, dtype=torch.bfloat16):
        kv = []
        for i in range(self.cfg.num_hidden_layers):
            shape = (batch, self.cfg.layer_kv_heads(i), 0, self.cfg.layer_head_dim(i))
            kv.append((torch.zeros(shape, dtype=dtype), torch.zeros(shape, dtype=dtype)))
        return kv

    def forward_chunk(self, ids, start_pos: int, kv, return_logits: bool = True, last_token_only: bool = True):
        """Prefill one chunk ``ids [B, S]`` at absolute ``start_pos``; ``kv`` holds all earlier
        chunks' per-layer (K, V) ``[B, Hkv, start_pos, D]`` and is extended in place."""
        B, S = ids.shape
        q_pos = torch.arange(start_pos, start_pos + S)
        k_pos = torch.arange(0, start_pos + S)
        cs = {t: rope_cos_sin(q_pos, s.theta, s.head_dim, s.rotated_pairs) for t, s in self.rope_specs.items()}
        masks = {
            SLIDING: attention_mask(q_pos, k_pos, self.cfg.sliding_window),
            FULL: attention_mask(q_pos, k_pos, None),
        }
        x = self.embed_tokens(ids)
        for i, layer in enumerate(self.layers):
            t = layer.layer_type
            kp, vp = kv[i]
            x, k, v = layer(x, *cs[t], kp, vp, masks[t])
            kv[i] = (torch.cat([kp, k.to(kp.dtype)], 2), torch.cat([vp, v.to(vp.dtype)], 2))
        if not return_logits:
            return x
        h = self.norm(x[:, -1:] if last_token_only else x)
        return self.lm_head(h)

    def prefill(self, ids, chunk_size: int | None = None, **kw):
        """Full prefill, optionally chunked. Returns (logits, kv)."""
        S = ids.shape[1]
        chunk_size = chunk_size or S
        kv = self.empty_kv(ids.shape[0], dtype=self.embed_tokens.weight.dtype)
        outs = [self.forward_chunk(ids[:, s : s + chunk_size], s, kv, **kw) for s in range(0, S, chunk_size)]
        if kw.get("return_logits", True):
            return outs[-1], kv
        return torch.cat(outs, dim=1), kv
