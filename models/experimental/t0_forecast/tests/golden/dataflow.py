# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""CPU torch re-implementation of the t0 network in the device's decomposition (fp32, no ttnn).

reference : t0/forecaster.py (``_backbone``) and t0/torch/model/layers/*
It is the spec the TTNN modules follow and is checked against the reference at 1e-5 (``test_dataflow.py``):
    embed        relu(tok @ W1 + b1) @ W2 + tok @ Wr + (b2 + br) + type_emb[patch_type]
    time layer   q, k: per-head RMSNorm (gain) then RoPE as ``t * cos + (t @ R) * sin`` with xPos folded
                 into separate q / k tables; SDPA causal (or the structured mask) with scale 1/sqrt(128)
    group layer  one row per series: x + (xn @ Wv + bv) @ Wo + bo (softmax over one key is 1);
                 else SDPA over the G rows of each series at every patch
    MLP          silu(gate) * up with gate the FIRST half of w1
    read-out     everything after the last time attention is per token (per patch for group layers),
                 so the stack is cut to the read-out patches there
    head         decoder ResidualBlock (ReLU); monotone head as ``[raw_0, softplus(raw_1..)] @ U`` (U upper
                 triangular ones) per step, i.e. one block-diagonal (672 x 672) matmul
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from models.experimental.t0_forecast.tt.host import T0HostInputs, monotone_matrix, pair_rotation
from models.experimental.t0_forecast.tt.weights import (
    TIME,
    AttentionWeights,
    MlpWeights,
    ResidualBlockWeights,
    T0Weights,
)


def rms(x: Tensor, gain: Tensor, eps: float) -> Tensor:
    return x / torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps) * gain


def residual_block(w: ResidualBlockWeights, x: Tensor) -> Tensor:
    hidden = torch.relu(x @ w.hidden_w.T + w.hidden_b)
    return hidden @ w.output_w.T + x @ w.residual_w.T + (w.output_b + w.residual_b)


class Dataflow:
    """``network(prepared) -> [R, n_readout, patch_size, Q]`` in asinh space; ``trace`` collects intermediates."""

    def __init__(self, weights: T0Weights, *, readout_cut: bool = True):
        self.w = weights
        self.cfg = weights.config
        self.readout_cut = readout_cut
        self.first, self.cumulative = monotone_matrix(self.cfg.n_quantiles, self.cfg.patch_size)
        self.rotation = pair_rotation(self.cfg.head_dim)
        kinds = [layer.kind for layer in weights.layers]
        self.last_time = max(i for i, k in enumerate(kinds) if k == TIME)

    # --- modules -----------------------------------------------------------------------------------------

    def embed(self, p: T0HostInputs) -> Tensor:
        return residual_block(self.w.patch_encoder, p.tokens) + self.w.type_embeddings[p.patch_type]

    def heads(self, t: Tensor) -> Tensor:
        """``[..., S, d] -> [..., H, S, hd]``"""
        h, hd = self.cfg.num_heads, self.cfg.head_dim
        return t.reshape(*t.shape[:-1], h, hd).transpose(-3, -2)

    def merge(self, t: Tensor) -> Tensor:
        return t.transpose(-3, -2).reshape(*t.shape[:-3], t.shape[-2], -1)

    def rope(self, t: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        return t * cos + (t @ self.rotation) * sin

    def attend(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None, causal: bool) -> Tensor:
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.cfg.head_dim)
        if causal:
            n = scores.shape[-1]
            mask = torch.tril(torch.ones(n, n, dtype=torch.bool))
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        return torch.softmax(scores, dim=-1) @ v

    def qkv(self, a: AttentionWeights, xn: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        eps = self.cfg.norm_eps
        q, k, v = (self.heads(xn @ w.T + b) for w, b in (a.part(0), a.part(1), a.part(2)))
        return rms(q, a.q_norm, eps), rms(k, a.k_norm, eps), v

    def time_attention(self, a: AttentionWeights, x: Tensor, p: T0HostInputs) -> Tensor:
        """``[R, S, d]`` -> attention output (before the residual add)."""
        xn = rms(x, a.norm, self.cfg.norm_eps)
        q, k, v = self.qkv(a, xn)
        cos_q, sin_q, cos_k, sin_k = p.rope
        q, k = self.rope(q, cos_q, sin_q), self.rope(k, cos_k, sin_k)
        mask = None if p.time_mask is None else p.time_mask[:, None]
        ctx = self.attend(q, k, v, mask, causal=p.time_mask is None)
        return self.merge(ctx) @ a.wo.T + a.bo

    def group_attention(self, a: AttentionWeights, x: Tensor, p: T0HostInputs, group_mask: Tensor | None) -> Tensor:
        """``[R, S', d]`` -> attention output; ``group_mask`` ``[S', R / G, G, G]`` for the patches in ``x``."""
        xn = rms(x, a.norm, self.cfg.norm_eps)
        g = p.group_size
        if g == 1:
            wv, bv = a.part(2)
            return (xn @ wv.T + bv) @ a.wo.T + a.bo
        r, s, d = xn.shape
        slot = xn.reshape(r // g, g, s, d).permute(2, 0, 1, 3)  # [S, B, G, d]
        q, k, v = self.qkv(a, slot)  # [S, B, H, G, hd]
        mask = None if group_mask is None else group_mask[:, :, None]
        ctx = self.merge(self.attend(q, k, v, mask, causal=False))  # [S, B, G, d]
        out = ctx @ a.wo.T + a.bo
        return out.permute(1, 2, 0, 3).reshape(r, s, d)

    def mlp(self, m: MlpWeights, x: Tensor) -> Tensor:
        xn = rms(x, m.norm, self.cfg.norm_eps)
        gate, up = (xn @ m.w1.T + m.b1).chunk(2, dim=-1)
        return (F.silu(gate) * up) @ m.w2.T + m.b2

    def head(self, x: Tensor) -> Tensor:
        """Read-out hidden states ``[R, n, d]`` -> ``[R, n, ps, Q]``: out_norm, decoder, monotone head."""
        raw = residual_block(self.w.decoder, rms(x, self.w.out_norm, self.cfg.norm_eps))
        y = raw * self.first + F.softplus(raw) * (1 - self.first)
        q = y @ self.cumulative
        return q.reshape(*q.shape[:-1], self.cfg.patch_size, self.cfg.n_quantiles)

    # --- the network -------------------------------------------------------------------------------------

    def __call__(self, p: T0HostInputs, trace: dict | None = None) -> Tensor:
        lo, n = p.readout_start, p.readout_count
        group_mask = p.group_mask
        x = self.embed(p)
        if trace is not None:
            trace["patch_encoder"] = x
        cut = False
        for i, layer in enumerate(self.w.layers):
            a = layer.attention
            if a.kind == TIME:
                x = x + self.time_attention(a, x, p)
            else:
                x = x + self.group_attention(a, x, p, group_mask)
            if self.readout_cut and i == self.last_time and not cut:
                x = x[:, lo : lo + n]
                group_mask = None if group_mask is None else group_mask[lo : lo + n]
                cut = True
            x = x + self.mlp(layer.mlp, x)
            if trace is not None:
                trace[f"layer_{i}"] = x
        if not cut:
            x = x[:, lo : lo + n]
        return self.head(x)
