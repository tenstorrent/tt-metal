# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Folded fp32 torch Toto 2.0 in the device's decomposition: the dataflow the TTNN model implements.

Weights come from ``tt/weights.py`` (u-muP constants folded) and inputs from ``tt/preprocessing.py``, the same
code the device model uses. The forward mirrors the device step by step:

    x = silu(tok W1 + b1) W2 + tok Ws + b                        rows (R, S_pad, d), pads after the real tokens
    time layer: xn = RMSNorm(x); q, k, v = heads(xn Wqkv + b)
                q, k = q cos_q + rot(q) sin_q, k cos_k + rot(k) sin_k     (RoPE + xPos tables, identity on 32-63)
                a = SDPA(q, k, v, 1/64, causal or the group mask);  x = beta_a x + a Wo + bo
                (last time layer: keep only the read-out tokens from here on)
                xn = RMSNorm(x); x = beta_m x + (g * silu(u)) Wfc2,  [g, u] = xn Wfc1
    variate layer: V = 1 -> a = (xn Wv + bv) Wo + bo;  else SDPA over the V variates of each token slot
    head(RMSNorm(x)) at the read-out tokens -> (9, B, V, h, 32)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from models.experimental.toto2_forecast.tt.positions import apply_rope
from models.experimental.toto2_forecast.tt.preprocessing import Toto2HostInputs
from models.experimental.toto2_forecast.tt.weights import LayerWeights, ResidualMLPWeights, Toto2Weights


def rms_norm(x: Tensor, eps: float) -> Tensor:
    return x / (x.float().pow(2).mean(-1, keepdim=True) + eps).sqrt().to(x.dtype)


def residual_mlp(w: ResidualMLPWeights, x: Tensor) -> Tensor:
    return F.silu(F.linear(x, w.w1, w.b1)) @ w.w2.t() + x @ w.ws.t() + w.b


class FoldedToto2:
    def __init__(self, weights: Toto2Weights, dtype=torch.float32, fold_betas: bool = False):
        self.w = weights.fold_residual_betas() if fold_betas else weights
        self.c = weights.config
        self.dtype = dtype

    def _cast(self, t):
        return t.to(self.dtype)

    def embed(self, tokens: Tensor) -> Tensor:
        return residual_mlp(self._mlp(self.w.patch_proj), self._cast(tokens))

    def _mlp(self, w: ResidualMLPWeights) -> ResidualMLPWeights:
        if self.dtype == torch.float32:
            return w
        return ResidualMLPWeights(*(self._cast(t) for t in (w.w1, w.b1, w.w2, w.ws, w.b)))

    def _heads(self, t: Tensor) -> Tensor:
        """``(N, S, H * Dh)`` -> ``(N, H, S, Dh)``."""
        n, s, _ = t.shape
        return t.reshape(n, s, self.c.num_heads, self.c.head_dim).transpose(1, 2)

    def time_attention(self, lw: LayerWeights, x: Tensor, host: Toto2HostInputs):
        """``x`` ``(R, S_pad, d)`` -> attention residual output, same shape."""
        d = self.c.d_model
        r, sp, _ = x.shape
        xn = rms_norm(x, self.c.norm_eps if lw.eps_attn is None else lw.eps_attn)
        qkv = F.linear(xn, self._cast(lw.w_qkv), self._cast(lw.b_qkv))
        q, k, v = (self._heads(qkv[..., i * d : (i + 1) * d]) for i in range(3))
        rope = host.rope
        s = host.seq_len

        def table(t, identity):
            t = self._cast(t)
            if sp == s:
                return t
            pad = torch.full((sp - s, t.shape[-1]), identity, dtype=t.dtype)
            return torch.cat([t, pad])

        q = apply_rope(q, table(rope.cos_q, 1.0), table(rope.sin_q, 0.0))
        k = apply_rope(k, table(rope.cos_k, 1.0), table(rope.sin_k, 0.0))
        if host.time_mask is None:
            ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0 / self.c.head_dim)
        else:
            mask = torch.tril(torch.ones(sp, sp, dtype=torch.bool)).expand(r, sp, sp).clone()
            mask[:, :s, :] = False
            mask[:, :s, :s] = host.time_mask
            ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None], scale=1.0 / self.c.head_dim)
        ctx = ctx.transpose(1, 2).reshape(r, sp, d)
        return lw.beta_attn * x + F.linear(ctx, self._cast(lw.w_o), self._cast(lw.b_o))

    def mlp(self, lw: LayerWeights, x: Tensor) -> Tensor:
        xn = rms_norm(x, self.c.norm_eps if lw.eps_mlp is None else lw.eps_mlp)
        g, u = F.linear(xn, self._cast(lw.w_fc1)).chunk(2, dim=-1)
        return lw.beta_mlp * x + F.linear(g * F.silu(u), self._cast(lw.w_fc2))

    def variate_attention(self, lw: LayerWeights, x: Tensor, host: Toto2HostInputs, slots: slice) -> Tensor:
        """``x`` ``(R, n, d)`` at token ``slots`` -> same shape."""
        d = self.c.d_model
        r, n, _ = x.shape
        b, v_count = host.batch, host.variates
        xn = rms_norm(x, self.c.norm_eps if lw.eps_attn is None else lw.eps_attn)
        w_o, b_o = self._cast(lw.w_o), self._cast(lw.b_o)
        if v_count == 1:
            v = F.linear(xn, self._cast(lw.w_qkv[2 * d :]), self._cast(lw.b_qkv[2 * d :]))
            return lw.beta_attn * x + F.linear(v, w_o, b_o)
        qkv = F.linear(xn, self._cast(lw.w_qkv), self._cast(lw.b_qkv))
        # (B * V, n, 3d) -> (B * n, V, 3d): attend across the variates of one item at one token slot.
        qkv = qkv.reshape(b, v_count, n, 3 * d).transpose(1, 2).reshape(b * n, v_count, 3 * d)
        q, k, v = (self._heads(qkv[..., i * d : (i + 1) * d]) for i in range(3))
        mask = None
        if host.variate_mask is not None:
            mask = host.variate_mask[:, slots].reshape(b * n, 1, v_count, v_count)
        ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0 / self.c.head_dim)
        ctx = ctx.transpose(1, 2).reshape(b, n, v_count, d).transpose(1, 2).reshape(r, n, d)
        return lw.beta_attn * x + F.linear(ctx, w_o, b_o)

    def head(self, x: Tensor) -> Tensor:
        """``(R, n, d)`` read-out tokens -> ``(R, n, 288)``."""
        return residual_mlp(
            self._mlp(self.w.head), rms_norm(x, self.c.norm_eps if self.w.head_eps is None else self.w.head_eps)
        )

    def forward(self, host: Toto2HostInputs, *, padded_seq: int | None = None, cut: bool = True, capture=None):
        """Head output ``(9, B, V, h, 32)``. Rows run padded to ``padded_seq`` tokens (pads after the real ones)
        until the last time layer. ``capture`` (a dict) receives ``embed`` and ``layer_<i>`` in the reference's
        layouts; with ``cut`` the layers from the last time layer on hold only the read-out tokens."""
        s = host.seq_len
        sp = padded_seq or s
        start, count = host.readout
        tokens = host.tokens
        if sp > s:
            tokens = torch.cat([tokens, torch.zeros(tokens.shape[0], sp - s, tokens.shape[-1])], dim=1)
        x = self.embed(tokens)
        if capture is not None:
            capture["embed"] = x[:, :s]
        layers = self.w.layers
        last_time = max(i for i, lw in enumerate(layers) if not lw.is_variate)
        assert all(lw.is_variate for lw in layers[last_time + 1 :]), "variate layers must follow the time layers"
        slots = slice(0, s)
        for i, lw in enumerate(layers):
            if lw.is_variate:
                x = self.variate_attention(lw, x, host, slots)
            else:
                x = self.time_attention(lw, x, host)
                if i == last_time:
                    slots = slice(start, start + count) if cut else slice(0, s)
                    x = x[:, slots]
            x = self.mlp(lw, x)
            if capture is not None:
                capture[f"layer_{lw.index}"] = self._reference_layout(x[:, :s], host, lw.is_variate)
        if x.shape[1] != count:
            x = x[:, start : start + count]
        out = self.head(x)  # (R, h, 32 * Q), columns step-major
        out = out.reshape(host.batch, host.variates, count, 32, self.c.num_quantiles)
        return out.permute(4, 0, 1, 2, 3)

    @staticmethod
    def _reference_layout(x: Tensor, host: Toto2HostInputs, is_variate: bool) -> Tensor:
        """Time layers ``(B * V, S, d)``; the variate layer ``(B * S, V, d)``."""
        if not is_variate:
            return x
        r, n, d = x.shape
        return x.reshape(host.batch, host.variates, n, d).transpose(1, 2).reshape(host.batch * n, host.variates, d)
