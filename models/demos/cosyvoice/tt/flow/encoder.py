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


def _linear_fused(device, bag, names, dtype, weights_dtype=None, scales=None):
    """Several sub-linears over the same input, concatenated into one `[d_in, sum(d_out)]`.

    q, k and v are three projections of the *same* activation, so they can be one
    matmul over a wider weight. The concatenation happens on the host, once, at
    construction -- the device sees a single weight and never learns there were three.

    **It pays where the matmuls are large and not where they are not**, which is worth
    stating because the op count alone predicts the opposite. In the flow estimator
    (T ~ 600, batch 2, 64 blocks x 10 Euler steps) this took the stage from 1.075 s to
    0.818 s. In the AR decode step, where T = 1, it measured 8.29 -> 8.31 ms: a wash,
    because `split_query_key_value_and_split_heads` physically rearranges the fused
    row into three head-major tensors and at that size costs about what the two
    matmuls it removed did. Fewer ops is a proxy for cost, not the cost itself.

    A missing bias among present ones is filled with zeros rather than dropping the
    bias for all three, since the fused linear has to be all-or-nothing.

    `scales` optionally multiplies each sub-weight (and its bias) by a constant on the
    host. Folding attention's `1/sqrt(d_head)` into the q half this way deletes a
    device `multiply` per block: scaling `q` before the product is the same as scaling
    `q @ k^T` after it, because the scalar distributes over the matmul. It is exact in
    the sense that matters here -- the constant is applied once, in fp32, at load time,
    rather than once per block per step in bfloat16.
    """
    scales = scales or (1.0,) * len(names)
    ws, bs, widths = [], [], []
    for name, s in zip(names, scales):
        sub = bag.sub(name)
        w = sub.tensor("weight").t().contiguous()
        if s != 1.0:
            w = w * s
        ws.append(w)
        widths.append(w.shape[-1])
        b = sub.tensor("bias").reshape(1, 1, -1) if sub.has("bias") else None
        if b is not None and s != 1.0:
            b = b * s
        bs.append(b)

    weight = ttnn.from_torch(
        torch.cat(ws, dim=-1), dtype=weights_dtype or dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    bias = None
    if any(b is not None for b in bs):
        parts = [b if b is not None else torch.zeros(1, 1, w) for b, w in zip(bs, widths)]
        bias = ttnn.from_torch(torch.cat(parts, dim=-1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    return weight, bias


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
        self.wqkv, self.bqkv = _linear_fused(device, bag, ("linear_q", "linear_k", "linear_v"), dtype, weights_dtype)
        self.wo, self.bo = _linear(device, bag, "linear_out", dtype, weights_dtype)
        self.wp, _ = _linear(device, bag, "linear_pos", dtype, weights_dtype)  # bias=False upstream
        # [h, d_k] -> [1, h, 1, d_k] so it broadcasts over time
        self.bias_u = ttnn.from_torch(
            bag.tensor("pos_bias_u").reshape(1, n_head, 1, d_k), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        self.bias_v = ttnn.from_torch(
            bag.tensor("pos_bias_v").reshape(1, n_head, 1, d_k), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        # Off by default; only the fixed-width decode path turns it on. See `_pos_proj`.
        self.cache_pos_proj = False
        # id(pos_emb) -> (pos_emb, pt). The tensor itself is kept so that its `id`
        # cannot be recycled into a stale hit.
        self._pt_cache: dict = {}

    def _pos_proj(self, pos_emb, b):
        """`linear_pos`, head-split and transposed -- cached on the `pos_emb` object.

        **This is the largest matmul in the layer by two orders of magnitude, and on
        the decode path it recomputes an identical result every token.** `pos_emb` is
        `[B, 2*key_len - 1, d_model]`, so at `max_len = 256` this projects 511 rows
        through `[1024, 1024]` -- about 536 MFLOP -- while q, k and v each project a
        single row, about 1 MFLOP apiece. Roughly 97 % of the layer's arithmetic sits
        in the one branch that does not depend on the token being decoded.

        It is loop-invariant because `TtARDecoder.positional()` hands back the *same
        cached tensor* on every step: the window is a function of `max_len`, which is
        fixed for an utterance. Caching `pt` therefore removes a `linear`, a `reshape`
        and a `permute` per layer per token, and the `linear` is the expensive one.

        The cache is keyed on `id(pos_emb)` rather than on its width, because width
        alone would be a lie: two different callers can legitimately pass different
        `[B, N, d_model]` windows. Keying on identity makes a hit mean "the very
        tensor whose projection this is". The entry holds a reference to `pos_emb`
        precisely so that CPython cannot recycle its `id` into a stale hit.

        Allocation happens on first call, which for the traced path is a warm-up pass
        -- outside `begin_trace_capture`, like the weights, so the replay only reads.

        **Caching is opt-in, and deliberately so.** It pays only where the same window
        is reused, and it is unsafe wherever entries would have to be evicted: the
        traced decode step holds a device pointer to `pt` for the life of the trace,
        so freeing one to make room would be a use-after-free that shows up as
        corrupted attention several hundred tokens later. The two callers differ:

        * `TtARDecoder`'s fixed-width path always passes `positional(max_len)` -- one
          window for the whole utterance, so the cache holds exactly one entry and
          never needs to evict. It calls `enable_pos_proj_cache()`.
        * The flow encoder's window is `2*T - 1` for the utterance's own `T`, and it
          runs once per utterance. Caching would buy nothing and leak an entry per
          distinct length, so it stays off.

        **Ownership follows the flag.** With the cache on, `pt` belongs to the cache
        and the caller must leave it alone; with it off, `pt` is a fresh tensor the
        caller must free. `forward_cached` branches on `self.cache_pos_proj` to do
        exactly that, and getting it backwards leaks a megabyte per layer per token.
        """
        if not self.cache_pos_proj:
            p = self._heads(ttnn.linear(pos_emb, self.wp, compute_kernel_config=self.cc), b, pos_emb.shape[1])
            pt = ttnn.permute(p, (0, 1, 3, 2))
            ttnn.deallocate(p)
            return pt
        hit = self._pt_cache.get(id(pos_emb))
        if hit is not None:
            return hit[1]
        p = self._heads(ttnn.linear(pos_emb, self.wp, compute_kernel_config=self.cc), b, pos_emb.shape[1])
        pt = ttnn.permute(p, (0, 1, 3, 2))
        ttnn.deallocate(p)
        self._pt_cache[id(pos_emb)] = (pos_emb, pt)
        return pt

    def _heads(self, x, b, t):
        """[B, T, d_model] -> [B, h, T, d_k].

        **At `T == 1` the permute is a relabelling and is skipped.** `[b, 1, h, d_k]`
        and `[b, h, 1, d_k]` enumerate the same elements in the same order when the
        time axis has extent 1, so the reshape alone reaches the target shape and
        the permute has nothing left to move.

        This is a decode-path optimisation with a measured motive rather than a
        tidiness one. Counting the ops in one decode step (`scripts/count_decode_ops
        .py`) puts `reshape` and `permute` together at **196 of 636 ops -- 31 %**,
        more than every `linear` and `matmul` combined, on a step that `bfloat8_b`
        weights showed to be per-op-bound rather than bandwidth-bound. Attention
        calls this three times per layer for q, k and v.

        The positional branch still permutes: its length is `2 * key_len - 1`, not 1.
        """
        if t == 1:
            return ttnn.reshape(x, (b, self.h, 1, self.d_k))
        x = ttnn.reshape(x, (b, t, self.h, self.d_k))
        return ttnn.permute(x, (0, 2, 1, 3))

    @staticmethod
    def rel_shift(x, b, h, t1, n):
        """Skew the [B, h, T1, N] positional scores down to [B, h, T1, N//2+1].

        Pad a zero column, reinterpret the last two axes transposed, drop the
        first row, reinterpret back, keep the left half. Every step is a reshape
        or a slice -- no gather op needed.

        **At `t1 == 1` the whole sequence collapses to that final slice.** With one
        query row the pad-and-drop is its own inverse: prepending a zero to a length
        `n` row and then dropping the first element of the `(n+1, 1)` reinterpretation
        returns the original `n` elements in the original order, so only the trailing
        `[..., : n // 2 + 1]` slice survives. Seven ops become one, on a path that
        runs once per layer per generated token.

        This is a `t1 == 1` identity, not a general one -- it is exactly false for
        `t1 >= 2`, where the skew genuinely permutes. `test_rel_shift_fast_path_only_
        holds_at_t1_1` asserts both halves of that, so the guard cannot quietly widen.
        """
        if t1 == 1:
            return ttnn.slice(x, [0, 0, 0, 0], [b, h, 1, n // 2 + 1])
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

    def forward_cached(self, x, pos_emb, mask=None, cache=None, return_cache=False, cache_free=False):
        """The same attention, with an optional `(k, v)` cache prepended.

        `cache` is a pair of `[B, h, cache_t, d_k]` tensors, kept unpacked rather
        than in the reference's `[1, h, cache_t, 2*d_k]` packing -- that packing
        exists to make its ONNX export a single tensor and buys nothing here.

        `cache=None, return_cache=True` is the first chunk of an autoregressive
        decode: nothing to prepend, but the k/v it computes must be kept. The two
        flags are separate for exactly that case.

        Returns `(output, (k, v) | None)`, where the returned k/v span cache plus
        this chunk and become the caller's to free.

        **`cache_free` switches the cache to a time-major `[B, cache_t, h, d_k]`
        layout**, and it is worth a great deal on the decode path. `TILE_LAYOUT` tiles
        only the last two dimensions, so in the default `[B, h, T, d_k]` the time axis
        is *tiled* -- and appending one row to it re-tiles the whole buffer. Measured on
        a `[1, 16, 256, 64]` cache:

            slice + concat on the tiled time axis   207.2 us
            slice + concat on a free time axis       19.7 us
            permute back to [B, h, T, d_k]           13.9 us

        So paying a permute to do the append on a free axis is **6.2x cheaper** than
        appending on the tiled one. The returned k/v are then in the free layout too --
        the caller writes those straight back to its buffers, and nothing else about the
        attention changes: same shapes into the matmuls, same geometry, same trace.
        """
        b, t, _ = x.shape
        tp = pos_emb.shape[1]

        # One matmul for all three projections, then one op to split them and lay out
        # the heads -- two ops where the obvious form is six. See `_linear_fused`.
        qkv = ttnn.linear(x, self.wqkv, bias=self.bqkv, compute_kernel_config=self.cc)
        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(qkv, num_heads=self.h, transpose_key=False)
        ttnn.deallocate(qkv)
        pt = self._pos_proj(pos_emb, b)  # cached, and NOT ours to deallocate

        kv_free = None
        if cache is not None and cache_free:
            # Append on the free time axis, then permute once. `k` arrives from the
            # split as [B, h, 1, d_k]; the two small permutes cost a few microseconds
            # between them against the ~187 us the tiled append would cost.
            ck, cv = cache  # each [B, cache_t, h, d_k]
            k1 = ttnn.permute(k, (0, 2, 1, 3))  # -> [B, 1, h, d_k]
            v1 = ttnn.permute(v, (0, 2, 1, 3))
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            k_free = ttnn.concat([ck, k1], dim=1)
            v_free = ttnn.concat([cv, v1], dim=1)
            ttnn.deallocate(k1)
            ttnn.deallocate(v1)
            k = ttnn.permute(k_free, (0, 2, 1, 3))  # -> [B, h, T, d_k] for the matmuls
            v = ttnn.permute(v_free, (0, 2, 1, 3))
            kv_free = (k_free, v_free)
        elif cache is not None:
            ck, cv = cache
            k_full = ttnn.concat([ck, k], dim=2)
            v_full = ttnn.concat([cv, v], dim=2)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            k, v = k_full, v_full

        qu = ttnn.add(q, self.bias_u)
        qv = ttnn.add(q, self.bias_v)
        ttnn.deallocate(q)

        # `transpose_b` folds the [B, h, T, d_k] -> [B, h, d_k, T] permute into the
        # matmul, which is one fewer op per layer per token and one fewer full copy
        # of the key block.
        ac = ttnn.matmul(qu, k, transpose_b=True, compute_kernel_config=self.cc)
        ttnn.deallocate(qu)

        bd = ttnn.matmul(qv, pt, compute_kernel_config=self.cc)
        ttnn.deallocate(qv)
        if not self.cache_pos_proj:
            ttnn.deallocate(pt)  # ours only when uncached -- see `_pos_proj`
        # `if matrix_ac.shape != matrix_bd.shape` upstream. With a KV cache the
        # comparison is against the *attention key size*, not the chunk length --
        # a one-token decode step still needs the skew.
        if bd.shape[-1] != ac.shape[-1]:
            bd = self.rel_shift(bd, b, self.h, t, tp)

        raw = ttnn.add(ac, bd)
        ttnn.deallocate(ac)
        ttnn.deallocate(bd)

        if mask is not None and mask.shape[-2] == 1:
            # `softmax(raw * scale + mask)` in one op instead of three.
            #
            # The shape guard is not defensive programming, it is the op's contract:
            # `scale_mask_softmax` accepts only a `[B, 1, 1, W]` padding mask and
            # raises `TT_FATAL ... softmax_device_operation.cpp:507` on a square
            # `[B, 1, T, T]` causal one -- with or without `is_causal_mask`. That is
            # exactly the split between this model's two paths: prefill and the text
            # encoder pass a causal mask and take the explicit branch, while a decode
            # step's mask is `[1, 1, 1, max_len]` and is already the wanted form.
            # So the fusion lands on the path that runs once per token and skips the
            # one that runs once per utterance, which is the right way round.
            attn = ttnn.scale_mask_softmax(raw, self.scale, mask)
        else:
            scaled = ttnn.multiply(raw, self.scale)
            if mask is not None:
                masked = ttnn.add(scaled, mask)  # additive, broadcast over [B, 1|h, T, T_key]
                ttnn.deallocate(scaled)
                scaled = masked
            attn = ttnn.softmax(scaled, dim=-1)
            ttnn.deallocate(scaled)
        ttnn.deallocate(raw)

        ctx = ttnn.matmul(attn, v, compute_kernel_config=self.cc)  # [B, h, T, d_k]
        ttnn.deallocate(attn)
        if t == 1:
            ctx = ttnn.reshape(ctx, (b, 1, self.h * self.d_k))  # see _heads: permute is a no-op at T=1
        else:
            ctx = ttnn.permute(ctx, (0, 2, 1, 3))
            ctx = ttnn.reshape(ctx, (b, t, self.h * self.d_k))
        out = ttnn.linear(ctx, self.wo, bias=self.bo, compute_kernel_config=self.cc)
        ttnn.deallocate(ctx)

        if kv_free is not None:
            # The permuted [B, h, T, d_k] copies were only needed for the matmuls above;
            # what the caller keeps is the free-layout pair its buffers are in.
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            if not return_cache:
                for tns in kv_free:
                    ttnn.deallocate(tns)
                return out, None
            return out, kv_free

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
