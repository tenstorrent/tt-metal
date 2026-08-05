# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Conformer encoder with ESPnet relative-position attention.

The flow encoder is 6 pre-norm blocks, 8 heads, d=512, FFN 2048. Despite the
"Conformer" name this checkpoint sets `use_cnn_module: False` and
`macaron_style: False`, so each block is just

    x = x + self_attn(norm_mha(x), pos_emb)
    x = x + feed_forward(norm_ff(x))          # ff_scale == 1.0

with a final `after_norm` because `normalize_before` is True. The same structure
serves the LLM's text encoder (6 blocks, 16 heads, d=1024) and its AR decoder
(14 blocks), so this module is the reusable piece of P3 and P4 both.

Attention (see tests/pcc/test_rel_pos_attention.py, which verifies this against a
captured layer bit-exactly):

    matrix_ac = (q + pos_bias_u) @ k^T
    matrix_bd = rel_shift( (q + pos_bias_v) @ p^T )
    scores    = (matrix_ac + matrix_bd) / sqrt(d_k)

`rel_shift` is the awkward part on device: it is a *skew* of the score matrix,
done by padding a column, reinterpreting the last two axes transposed, dropping a
row and slicing. In tile layout that is a strided gather rather than an
elementwise op, which is why `03_plan.md` P5 rates a native rel-pos SDPA as its
high-risk item. Here it is composed from concat + reshape + slice, all of which
TTNN has.

Stage 1 computes the positional term explicitly, as `02_plan.md` §3.3 prescribes.
Folding it into an SDPA `attn_mask` is a Stage 3 change, and the identity that
makes it legal is already proven -- with the caveat that the bias must be
pre-divided by sqrt(d_k), since SDPA scales before adding.
"""
from __future__ import annotations

import math

import torch

import ttnn


def _linear(device, bag, name, dtype):
    """Weights arrive as torch [out, in]; ttnn.linear wants [in, out]."""
    sub = bag.sub(name)
    w = ttnn.from_torch(sub.tensor("weight").t().contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    b = None
    if sub.has("bias"):
        b = ttnn.from_torch(sub.tensor("bias").reshape(1, 1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return w, b


def _layernorm_weights(device, bag, name, dtype):
    """LayerNorm gamma/beta must be [1, C] in TILE_LAYOUT, not 1-D ROW_MAJOR.

    A 1-D ROW_MAJOR tensor of shape [C] has padded_shape[-1] == C, and
    ttnn.layer_norm requires gamma.padded_shape[-1] == 32. Reshaping to (1, C)
    and tilizing is what satisfies it -- the same trap CLAUDE.md records for the
    TTM-R1 bring-up.
    """
    sub = bag.sub(name)
    g = ttnn.from_torch(sub.tensor("weight").reshape(1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(sub.tensor("bias").reshape(1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return g, b


class TtRelPosAttention:
    """ESPnet RelPositionMultiHeadedAttention, explicit-matmul form."""

    def __init__(self, device, bag, n_head: int, d_k: int, dtype=ttnn.bfloat16):
        self.device, self.h, self.d_k, self.dtype = device, n_head, d_k, dtype
        self.scale = 1.0 / math.sqrt(d_k)
        self.wq, self.bq = _linear(device, bag, "linear_q", dtype)
        self.wk, self.bk = _linear(device, bag, "linear_k", dtype)
        self.wv, self.bv = _linear(device, bag, "linear_v", dtype)
        self.wo, self.bo = _linear(device, bag, "linear_out", dtype)
        self.wp, _ = _linear(device, bag, "linear_pos", dtype)  # bias=False upstream
        # [h, d_k] -> [1, h, 1, d_k] so it broadcasts over time
        self.bias_u = ttnn.from_torch(
            bag.tensor("pos_bias_u").reshape(1, n_head, 1, d_k), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.bias_v = ttnn.from_torch(
            bag.tensor("pos_bias_v").reshape(1, n_head, 1, d_k), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

    def _heads(self, x, b, t):
        """[B, T, d_model] -> [B, h, T, d_k]."""
        x = ttnn.reshape(x, (b, t, self.h, self.d_k))
        return ttnn.permute(x, (0, 2, 1, 3))

    @staticmethod
    def rel_shift(x, b, h, t1, n):
        """Skew the [B, h, T1, N] positional scores down to [B, h, T1, N//2+1].

        Pad a zero column, reinterpret the last two axes transposed, drop the
        first row, reinterpret back, keep the left half. Every step is a reshape
        or a slice -- no gather op needed.
        """
        zero = ttnn.zeros((b, h, t1, 1), dtype=x.dtype, layout=x.layout, device=x.device())
        padded = ttnn.concat([zero, x], dim=-1)  # [B, h, T1, N+1]
        ttnn.deallocate(zero)
        padded = ttnn.reshape(padded, (b, h, n + 1, t1))
        dropped = ttnn.slice(padded, [0, 0, 1, 0], [b, h, n + 1, t1])  # [B, h, N, T1]
        ttnn.deallocate(padded)
        back = ttnn.reshape(dropped, (b, h, t1, n))
        ttnn.deallocate(dropped)
        out = ttnn.slice(back, [0, 0, 0, 0], [b, h, t1, n // 2 + 1])
        ttnn.deallocate(back)
        return out

    def __call__(self, x, pos_emb, mask=None):
        """x: [B, T, d_model]; pos_emb: [B, 2T-1, d_model] -> [B, T, d_model]."""
        b, t, _ = x.shape
        tp = pos_emb.shape[1]

        q = self._heads(ttnn.linear(x, self.wq, bias=self.bq), b, t)
        k = self._heads(ttnn.linear(x, self.wk, bias=self.bk), b, t)
        v = self._heads(ttnn.linear(x, self.wv, bias=self.bv), b, t)
        p = self._heads(ttnn.linear(pos_emb, self.wp), b, tp)

        qu = ttnn.add(q, self.bias_u)
        qv = ttnn.add(q, self.bias_v)
        ttnn.deallocate(q)

        kt = ttnn.permute(k, (0, 1, 3, 2))
        ac = ttnn.matmul(qu, kt)
        ttnn.deallocate(qu)
        ttnn.deallocate(kt)

        pt = ttnn.permute(p, (0, 1, 3, 2))
        ttnn.deallocate(p)
        bd = ttnn.matmul(qv, pt)
        ttnn.deallocate(qv)
        ttnn.deallocate(pt)
        if tp != t:
            bd = self.rel_shift(bd, b, self.h, t, tp)

        scores = ttnn.multiply(ttnn.add(ac, bd), self.scale)
        ttnn.deallocate(ac)
        ttnn.deallocate(bd)

        if mask is not None:
            # mask is [B, 1, T] (1 = keep); make it [B, 1, 1, T] additive
            scores = ttnn.add(scores, mask)
        attn = ttnn.softmax(scores, dim=-1)
        ttnn.deallocate(scores)

        ctx = ttnn.matmul(attn, v)  # [B, h, T, d_k]
        ttnn.deallocate(attn)
        ttnn.deallocate(v)
        ctx = ttnn.permute(ctx, (0, 2, 1, 3))
        ctx = ttnn.reshape(ctx, (b, t, self.h * self.d_k))
        out = ttnn.linear(ctx, self.wo, bias=self.bo)
        ttnn.deallocate(ctx)
        return out


class TtConformerLayer:
    """One pre-norm block: attention then position-wise feed-forward."""

    def __init__(self, device, bag, meta, dtype=ttnn.bfloat16):
        self.device, self.dtype = device, dtype
        self.eps = meta["layer_norm_eps"]
        self.ff_scale = meta["ff_scale"]
        self.attn = TtRelPosAttention(device, bag.sub("self_attn"), meta["n_head"], meta["d_k"], dtype)
        self.w1, self.b1 = _linear(device, bag, "feed_forward.w_1", dtype)
        self.w2, self.b2 = _linear(device, bag, "feed_forward.w_2", dtype)
        self.g_mha, self.bt_mha = _layernorm_weights(device, bag, "norm_mha", dtype)
        self.g_ff, self.bt_ff = _layernorm_weights(device, bag, "norm_ff", dtype)

    def __call__(self, x, pos_emb, mask=None):
        h = ttnn.layer_norm(x, weight=self.g_mha, bias=self.bt_mha, epsilon=self.eps)
        a = self.attn(h, pos_emb, mask)
        ttnn.deallocate(h)
        x1 = ttnn.add(x, a)
        ttnn.deallocate(a)
        ttnn.deallocate(x)

        h = ttnn.layer_norm(x1, weight=self.g_ff, bias=self.bt_ff, epsilon=self.eps)
        f = ttnn.linear(h, self.w1, bias=self.b1)
        ttnn.deallocate(h)
        f = ttnn.silu(f)  # PositionwiseFeedForward uses SiLU here
        f2 = ttnn.linear(f, self.w2, bias=self.b2)
        ttnn.deallocate(f)
        if self.ff_scale != 1.0:
            f2 = ttnn.multiply(f2, self.ff_scale)
        out = ttnn.add(x1, f2)
        ttnn.deallocate(f2)
        ttnn.deallocate(x1)
        return out


class TtConformerEncoder:
    """The 6-block stack, plus the input projection and the trailing norm."""

    def __init__(self, device, bag, meta, dtype=ttnn.bfloat16):
        self.device, self.dtype, self.meta = device, dtype, meta
        self.xscale = math.sqrt(meta["d_model"])
        # embed.out is Sequential(Linear, LayerNorm, Dropout) for LinearNoSubsampling
        self.w_in, self.b_in = _linear(device, bag, "embed.out.0", dtype)
        self.g_in, self.bt_in = _layernorm_weights(device, bag, "embed.out.1", dtype)
        self.layers = [TtConformerLayer(device, bag.sub(f"encoders.{i}"), meta, dtype) for i in range(meta["n_layers"])]
        self.g_after, self.bt_after = _layernorm_weights(device, bag, "after_norm", dtype)

    def __call__(self, x, pos_emb, mask=None):
        """x: [B, T, input_size] -> [B, T, d_model]."""
        h = ttnn.linear(x, self.w_in, bias=self.b_in)
        h = ttnn.layer_norm(h, weight=self.g_in, bias=self.bt_in, epsilon=self.meta["layer_norm_eps"])
        # LinearNoSubsampling scales by sqrt(d_model) inside its positional encoding
        h = ttnn.multiply(h, self.xscale)
        for layer in self.layers:
            h = layer(h, pos_emb, mask)
        return ttnn.layer_norm(h, weight=self.g_after, bias=self.bt_after, epsilon=self.meta["layer_norm_eps"])

    # -- host reference, for isolating "identity wrong" from "op misbehaved" ---
    @staticmethod
    def torch_reference_attention(x, pos_emb, w, n_head, d_k, mask=None):
        """The attention layer in torch, matching test_rel_pos_attention.py."""
        from models.demos.cosyvoice.tests.pcc.test_rel_pos_attention import reference_scores  # noqa: E402

        b, t, _ = x.shape

        def proj(src, name):
            return torch.nn.functional.linear(src, w[f"{name}.weight"], w.get(f"{name}.bias"))

        q = proj(x, "linear_q").view(b, t, n_head, d_k)
        k = proj(x, "linear_k").view(b, t, n_head, d_k).transpose(1, 2)
        v = proj(x, "linear_v").view(b, t, n_head, d_k).transpose(1, 2)
        p = torch.nn.functional.linear(pos_emb, w["linear_pos.weight"])
        p = p.view(pos_emb.shape[0], -1, n_head, d_k).transpose(1, 2)
        scores = reference_scores(q, k, p, w["pos_bias_u"], w["pos_bias_v"], d_k)
        if mask is not None:
            m = mask.unsqueeze(1).eq(0) if mask.dim() == 3 else mask.eq(0)
            scores = scores.masked_fill(m, -float("inf"))
            attn = torch.softmax(scores, dim=-1).masked_fill(m, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, n_head * d_k)
        return torch.nn.functional.linear(ctx, w["linear_out.weight"], w["linear_out.bias"])
