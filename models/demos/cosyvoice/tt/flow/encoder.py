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

from ..hifigan.conv import accurate_compute_config


def espnet_rel_positional_encoding(size: int, d_model: int) -> torch.Tensor:
    """`EspnetRelPositionalEncoding.position_encoding(offset=0, size=T)`.

    Deterministic given (T, d_model), so it is generated rather than shipped in
    the weight export -- there is nothing learned here.

    The layout is the shifting trick from arXiv:1901.02860: positive positions
    reversed, then negative positions from index 1, concatenated to length
    2*max_len - 1, and then the middle 2T-1 window is taken. That reversal is
    what makes `rel_shift` in the attention meaningful; sampling it forwards
    instead produces a plausible-looking encoding with time running backwards.
    """
    max_len = size
    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
    pe_pos = torch.zeros(max_len, d_model)
    pe_neg = torch.zeros(max_len, d_model)
    pe_pos[:, 0::2] = torch.sin(position * div_term)
    pe_pos[:, 1::2] = torch.cos(position * div_term)
    pe_neg[:, 0::2] = torch.sin(-1 * position * div_term)
    pe_neg[:, 1::2] = torch.cos(-1 * position * div_term)

    pe_pos = torch.flip(pe_pos, [0]).unsqueeze(0)
    pe_neg = pe_neg[1:].unsqueeze(0)
    pe = torch.cat([pe_pos, pe_neg], dim=1)  # [1, 2*max_len - 1, d_model]

    mid = pe.size(1) // 2
    return pe[:, mid - size + 1 : mid + size]


def _linear(device, bag, name, dtype, weights_dtype=None):
    """Weights arrive as torch [out, in]; ttnn.linear wants [in, out].

    `weights_dtype` is separate from `dtype` so the matrix can be stored narrower
    than the activations flowing through it -- `bfloat8_b` weights with `bfloat16`
    activations halves what DRAM has to deliver. The bias stays at `dtype`: it is
    one row against a full matrix, so narrowing it buys no bandwidth and only
    costs accuracy.
    """
    sub = bag.sub(name)
    w = ttnn.from_torch(
        sub.tensor("weight").t().contiguous(),
        dtype=weights_dtype or dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
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

    def __init__(self, device, bag, n_head: int, d_k: int, dtype=ttnn.bfloat16, cc=None, weights_dtype=None):
        self.device, self.h, self.d_k, self.dtype = device, n_head, d_k, dtype
        # HiFi4 + fp32 accumulation: see F20 in the notes. The flow encoder is 6
        # blocks and the AR decoder is 14, and the decoder runs hundreds of times.
        self.cc = accurate_compute_config(device) if cc is None else cc
        self.scale = 1.0 / math.sqrt(d_k)
        self.wq, self.bq = _linear(device, bag, "linear_q", dtype, weights_dtype)
        self.wk, self.bk = _linear(device, bag, "linear_k", dtype, weights_dtype)
        self.wv, self.bv = _linear(device, bag, "linear_v", dtype, weights_dtype)
        self.wo, self.bo = _linear(device, bag, "linear_out", dtype, weights_dtype)
        self.wp, _ = _linear(device, bag, "linear_pos", dtype, weights_dtype)  # bias=False upstream
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
        # A zero column, made by zeroing a slice of `x` rather than with
        # `ttnn.zeros`. `ttnn.zeros(device=...)` is a host->device *write*, and
        # writes are illegal inside a trace capture -- "Writes are not supported
        # during trace capture", raised from `enqueue_write_tensor`, several frames
        # from this line. Multiplying a slice by zero is a pure device op and needs
        # no shape bookkeeping.
        zero = ttnn.multiply(ttnn.slice(x, [0, 0, 0, 0], [b, h, t1, 1]), 0.0)
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
        out, _ = self.forward_cached(x, pos_emb, mask=mask, cache=None)
        return out

    def forward_cached(self, x, pos_emb, mask=None, cache=None, return_cache=False):
        """The same attention, with an optional `(k, v)` cache prepended.

        `cache` is a pair of `[B, h, cache_t, d_k]` tensors, kept unpacked rather
        than in the reference's `[1, h, cache_t, 2*d_k]` packing -- that packing
        exists to make its ONNX export a single tensor and buys nothing here.

        `cache=None, return_cache=True` is the first chunk of an autoregressive
        decode: nothing to prepend, but the k/v it computes must be kept. The two
        flags are separate for exactly that case.

        Returns `(output, (k, v) | None)`, where the returned k/v span cache plus
        this chunk and become the caller's to free.
        """
        b, t, _ = x.shape
        tp = pos_emb.shape[1]

        q = self._heads(ttnn.linear(x, self.wq, bias=self.bq, compute_kernel_config=self.cc), b, t)
        k = self._heads(ttnn.linear(x, self.wk, bias=self.bk, compute_kernel_config=self.cc), b, t)
        v = self._heads(ttnn.linear(x, self.wv, bias=self.bv, compute_kernel_config=self.cc), b, t)
        p = self._heads(ttnn.linear(pos_emb, self.wp, compute_kernel_config=self.cc), b, tp)

        if cache is not None:
            ck, cv = cache
            k_full = ttnn.concat([ck, k], dim=2)
            v_full = ttnn.concat([cv, v], dim=2)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            k, v = k_full, v_full

        qu = ttnn.add(q, self.bias_u)
        qv = ttnn.add(q, self.bias_v)
        ttnn.deallocate(q)

        kt = ttnn.permute(k, (0, 1, 3, 2))
        ac = ttnn.matmul(qu, kt, compute_kernel_config=self.cc)
        ttnn.deallocate(qu)
        ttnn.deallocate(kt)

        pt = ttnn.permute(p, (0, 1, 3, 2))
        ttnn.deallocate(p)
        bd = ttnn.matmul(qv, pt, compute_kernel_config=self.cc)
        ttnn.deallocate(qv)
        ttnn.deallocate(pt)
        # `if matrix_ac.shape != matrix_bd.shape` upstream. With a KV cache the
        # comparison is against the *attention key size*, not the chunk length --
        # a one-token decode step still needs the skew.
        if bd.shape[-1] != ac.shape[-1]:
            bd = self.rel_shift(bd, b, self.h, t, tp)

        scores = ttnn.multiply(ttnn.add(ac, bd), self.scale)
        ttnn.deallocate(ac)
        ttnn.deallocate(bd)

        if mask is not None:
            # additive mask, already broadcastable to [B, 1|h, T, T_key]
            scores = ttnn.add(scores, mask)
        attn = ttnn.softmax(scores, dim=-1)
        ttnn.deallocate(scores)

        ctx = ttnn.matmul(attn, v, compute_kernel_config=self.cc)  # [B, h, T, d_k]
        ttnn.deallocate(attn)
        ctx = ttnn.permute(ctx, (0, 2, 1, 3))
        ctx = ttnn.reshape(ctx, (b, t, self.h * self.d_k))
        out = ttnn.linear(ctx, self.wo, bias=self.bo, compute_kernel_config=self.cc)
        ttnn.deallocate(ctx)

        if not return_cache:
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            return out, None
        return out, (k, v)


class TtConformerLayer:
    """One pre-norm block: attention then position-wise feed-forward."""

    def __init__(self, device, bag, meta, dtype=ttnn.bfloat16, cc=None):
        self.device, self.dtype = device, dtype
        self.cc = accurate_compute_config(device) if cc is None else cc
        self.eps = meta["layer_norm_eps"]
        self.ff_scale = meta.get("ff_scale", 1.0)
        # ConformerEncoder takes activation_type="swish" by default and
        # TransformerEncoder takes "relu"; cosyvoice.yaml overrides neither, so the
        # two stacks in one checkpoint genuinely differ. Read it, do not assume it.
        self.ffn_act = ttnn.relu if meta.get("ffn_activation", "silu") == "relu" else ttnn.silu
        self.attn = TtRelPosAttention(device, bag.sub("self_attn"), meta["n_head"], meta["d_k"], dtype, self.cc)
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
        f = ttnn.linear(h, self.w1, bias=self.b1, compute_kernel_config=self.cc)
        ttnn.deallocate(h)
        f = self.ffn_act(f)
        f2 = ttnn.linear(f, self.w2, bias=self.b2, compute_kernel_config=self.cc)
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
        self.cc = accurate_compute_config(device)
        self.xscale = math.sqrt(meta["d_model"])
        # subsampling.py pins eps=1e-5 on the embedding norm while encoder_layer.py
        # pins 1e-12 on the block norms -- genuinely two different values.
        self.embed_eps = meta.get("embed_norm_eps", 1e-5)
        self.has_relu = bool(meta.get("embed_has_relu", False))
        # embed.out is Sequential(Linear, LayerNorm, Dropout) for LinearNoSubsampling
        self.w_in, self.b_in = _linear(device, bag, "embed.out.0", dtype)
        self.g_in, self.bt_in = _layernorm_weights(device, bag, "embed.out.1", dtype)
        self.layers = [
            TtConformerLayer(device, bag.sub(f"encoders.{i}"), meta, dtype, self.cc) for i in range(meta["n_layers"])
        ]
        self.g_after, self.bt_after = _layernorm_weights(device, bag, "after_norm", dtype)

    def __call__(self, x, pos_emb, mask=None):
        """x: [B, T, input_size] -> [B, T, d_model].

        `mask` is an **additive** score bias, broadcastable to `[B, 1, T, T]`. The
        flow encoder passes None: its `static_chunk_size` is 0, so attention is
        full. The LLM's text encoder sets `static_chunk_size: 1`, which
        `subsequent_chunk_mask` turns into a plain causal mask -- same class, same
        weights layout, different attention pattern.
        """
        h = ttnn.linear(x, self.w_in, bias=self.b_in, compute_kernel_config=self.cc)
        h = ttnn.layer_norm(h, weight=self.g_in, bias=self.bt_in, epsilon=self.embed_eps)
        if self.has_relu:  # LegacyLinearNoSubsampling only
            h = ttnn.relu(h)
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
