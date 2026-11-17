# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""fp32 CPU reference for ONE Hy4-preview attention sublayer (HF ``hy_v4`` ``HYV4Attention``) given its top-k.

Input is the sublayer input ``hn`` (``input_layernorm`` output, i.e. after the iHC pre-mix), so this is
exactly what ttMLA computes: projections, DSA sparse MLA over the selected rows, the learnable per-head
sink, the elementwise sigmoid output gate and ``o_proj``. The math is the per-query-block pass of
``reference/hy4/model.py`` (#65919, the full-stack golden), in the absorbed (latent) form:

    scores[n, h, k] = (q_nope W_uk . latent_k + q_pe . k_pe_k) * softmax_scale       (masked by top-k, causal)
    probs           = softmax([scores | sink_h])[..., :-1]                           (sink: denominator only)
    o[n, h]         = (probs @ latent) W_uv^T                                         [n, H, v_head_dim]
    o              *= sigmoid(hn @ linear_gate^T).view(n, H, v_head_dim)              (elementwise gate)
    out             = o.reshape(n, H * v_head_dim) @ o_proj^T

RoPE is HF's half-split (``rotate_half``) on the raw checkpoint columns. Weights are HF names
(``q_a_proj.weight`` ... ``linear_gate.weight``, ``learnable_sink_param``). Import-light (torch only).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

F32 = torch.float32


@dataclass(frozen=True)
class Hy4AttnDims:
    num_heads: int = 64
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1e7

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def softmax_scale(self) -> float:  # plain RoPE, no YaRN mscale
        return self.qk_head_dim**-0.5

    @classmethod
    def from_config(cls, config) -> "Hy4AttnDims":
        return cls(
            num_heads=config.num_attention_heads,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
        )


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    x = x.to(F32)
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return y if weight is None else y * weight.to(F32)


def rope_cos_sin(positions: torch.Tensor, dim: int, theta: float):
    """``[N, dim/2]`` cos/sin, angle built in fp64 and rounded once (as the golden reference does)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
    ang = positions.to(torch.float64)[:, None] * inv_freq[None, :]
    return ang.cos().to(F32), ang.sin().to(F32)


def apply_rope_half_split(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """HF ``rotate_half`` RoPE on the last dim of ``x`` ``[N, ..., d]``; cos/sin ``[N, d/2]``."""
    shape = (cos.shape[0],) + (1,) * (x.dim() - 2) + (cos.shape[1],)
    c, s = cos.view(shape), sin.view(shape)
    h = x.shape[-1] // 2
    a, b = x[..., :h], x[..., h:]
    return torch.cat((a * c - b * s, b * c + a * s), dim=-1)


def sink_softmax(scores: torch.Tensor, sink: torch.Tensor | None) -> torch.Tensor:
    """Softmax over the last dim of ``scores`` ``[n, H, K]`` with one extra per-head logit ``sink`` ``[H]``
    that takes probability mass but has no value (HF ``hy_v4`` / gpt-oss sinks). ``sink=None``: plain softmax."""
    if sink is None:
        return torch.softmax(scores, dim=-1)
    n, H, _ = scores.shape
    logits = torch.cat([scores, sink.to(F32).view(1, H, 1).expand(n, H, 1)], dim=-1)
    return torch.softmax(logits - logits.amax(-1, keepdim=True), dim=-1)[..., :-1]


def mla_latents(hn: torch.Tensor, w: dict, dims: Hy4AttnDims, cos, sin):
    """Per-token stems: ``q_resid`` (``q_a_layernorm(q_a_proj(hn))``) and the cache line ``kvpe`` =
    ``kv_a_layernorm(latent) | rope(k_pe)`` (HF frame, half-split)."""
    L = dims.kv_lora_rank
    q_resid = rms_norm(hn @ w["q_a_proj.weight"].to(F32).T, w["q_a_layernorm.weight"], dims.rms_norm_eps)
    ckv = hn @ w["kv_a_proj_with_mqa.weight"].to(F32).T
    kvpe = torch.empty(hn.shape[0], L + dims.qk_rope_head_dim, dtype=F32)
    kvpe[:, :L] = rms_norm(ckv[:, :L], w["kv_a_layernorm.weight"], dims.rms_norm_eps)
    kvpe[:, L:] = apply_rope_half_split(ckv[:, L:], cos, sin)
    return q_resid, kvpe


def hy4_attention(
    hn: torch.Tensor,
    w: dict,
    topk: torch.Tensor,
    dims: Hy4AttnDims | None = None,
    *,
    use_sink: bool = True,
    use_gate: bool = True,
    block: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One Hy4 attention sublayer over a whole prompt (positions ``0..S-1``).

    ``hn`` ``[S, hidden]``; ``topk`` ``[S, K]`` absolute key positions per query, ``-1`` (or any value
    ``> query position``) = masked. Returns ``(out [S, hidden], kvpe [S, kv_lora + rope])`` in fp32.
    ``use_sink`` / ``use_gate`` switch the two Hy4 deltas off for diagnostics only."""
    dims = dims or Hy4AttnDims()
    hn = hn.to(F32)
    S = hn.shape[0]
    H, N, R, V, L = dims.num_heads, dims.qk_nope_head_dim, dims.qk_rope_head_dim, dims.v_head_dim, dims.kv_lora_rank
    cos, sin = rope_cos_sin(torch.arange(S), R, dims.rope_theta)
    q_resid, kvpe = mla_latents(hn, w, dims, cos, sin)

    wkv = w["kv_b_proj.weight"].to(F32).view(H, N + V, L)
    w_uk, w_uv = wkv[:, :N], wkv[:, N:]
    latent, k_pe = kvpe[:, :L], kvpe[:, L:]
    sink = w["learnable_sink_param"].to(F32) if use_sink else None
    wq_b = w["q_b_proj.weight"].to(F32)
    w_gate = w["linear_gate.weight"].to(F32) if use_gate else None
    w_o = w["o_proj.weight"].to(F32)
    out = torch.empty(S, w_o.shape[0], dtype=F32)
    topk = topk.long()
    for a in range(0, S, block):
        b = min(a + block, S)
        n = b - a
        q = (q_resid[a:b] @ wq_b.T).view(n, H, dims.qk_head_dim)
        q_nope, q_pe = q[..., :N], apply_rope_half_split(q[..., N:], cos[a:b], sin[a:b])
        q_lat = torch.einsum("nhd,hdc->nhc", q_nope, w_uk)
        idx = topk[a:b]
        pos = torch.arange(a, b).unsqueeze(1)
        valid = (idx >= 0) & (idx <= pos)
        safe = idx.clamp(min=0)
        lat_g, pe_g = latent[safe], k_pe[safe]  # [n, K, L], [n, K, R]
        scores = torch.einsum("nhc,nkc->nhk", q_lat, lat_g) + torch.einsum("nhr,nkr->nhk", q_pe, pe_g)
        scores = (scores * dims.softmax_scale).masked_fill(~valid.unsqueeze(1), float("-inf"))
        probs = sink_softmax(scores, sink)
        o = torch.einsum("nhc,hvc->nhv", torch.einsum("nhk,nkc->nhc", probs, lat_g), w_uv)
        if w_gate is not None:
            o = o * torch.sigmoid((hn[a:b] @ w_gate.T).view(n, H, V))
        out[a:b] = o.reshape(n, H * V) @ w_o.T
    return out, kvpe


def causal_topk_all(S: int, K: int) -> torch.Tensor:
    """``[S, K]`` top-k that selects every causal key while ``S <= K`` (rows padded with -1): the DSA
    selection whenever the prompt is no longer than ``index_topk``."""
    assert S <= K, f"S={S} > K={K}: the selection is no longer all causal keys"
    idx = torch.arange(K).unsqueeze(0).expand(S, K).clone()
    idx[idx > torch.arange(S).unsqueeze(1)] = -1
    return idx
