# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The autoregressive decoder: 14 pre-norm Transformer blocks with a KV cache.

Architecturally this is the flow encoder's block with two names changed
(`norm1`/`norm2` instead of `norm_mha`/`norm_ff`) and no macaron or conv module,
so `TtRelPosAttention` and the layer body are shared rather than duplicated.
The differences that matter are all at the edges:

**The input layer has a ReLU.** `input_layer: 'linear_legacy'` selects
`LegacyLinearNoSubsampling`, which is `Linear -> LayerNorm(eps=1e-5) -> Dropout ->
ReLU`. The plain `LinearNoSubsampling` the flow encoder uses stops at Dropout.
Nothing downstream would fail if the ReLU were missed -- the model would simply be
wrong. `embed_has_relu` in the exported meta records which one this checkpoint has.

**Two different LayerNorm epsilons.** `subsampling.py` pins `1e-5` on the
embedding norm; `encoder_layer.py` pins `1e-12` on the block norms. Both are
carried in the meta rather than assumed.

**The positional window follows the cache, not the chunk.** `forward_chunk`
recomputes `pos_emb = position_encoding(offset - cache_t1, size=cache_t1 + chunk)`,
discarding what the embedding produced. With `required_cache_size = -1` -- what
CosyVoice always passes -- the cache holds everything, so `offset == cache_t1` and
the offset term vanishes: the window is always the symmetric `2*key_size - 1` one.
That is what makes a single `espnet_rel_positional_encoding(key_size, d_model)`
correct at every step, and it is worth stating because it is only true for this
configuration.

**Prefill is causal; decode is not.** The prefill mask is `tril`, so the prompt
attends to itself autoregressively. A one-token decode step passes a `[1, 1, 1]`
all-true mask, and `forward_attention` slices it to the score width -- so it masks
nothing, which is right: every cached position is real history.
"""
from __future__ import annotations

import os

import torch

import ttnn

from ..flow.encoder import TtRelPosAttention, _layernorm_weights, _linear, decode_mask, espnet_rel_positional_encoding
from ..hifigan.conv import accurate_compute_config

NEG_INF = -1e9  # bfloat16-safe stand-in for -inf under softmax


def causal_bias(size: int, dtype=torch.float32) -> torch.Tensor:
    """Additive `[1, 1, T, T]` mask: 0 on and below the diagonal, very negative above."""
    keep = torch.tril(torch.ones(size, size, dtype=torch.bool))
    return torch.where(
        keep, torch.zeros(size, size, dtype=dtype), torch.full((size, size), NEG_INF, dtype=dtype)
    ).reshape(1, 1, size, size)


def _core_grid_from_env(var: str):
    """`ttnn.CoreGrid` from e.g. `COSYVOICE_FF2_GRID=8x2`, or None when unset.

    Off by default. The gain is real and architecture-independent in direction, but the
    best shape is not: 8x2 measured 1.50x on n300 and 1.98x on p150b, while 4x8 -- the
    same core count, transposed -- managed only 1.15x on n300. A default that is optimal
    on one part and mediocre on another is worse than an opt-in that says so.
    """
    val = os.environ.get(var, "").strip().lower()
    if not val:
        return None
    x, _, y = val.partition("x")
    return ttnn.CoreGrid(x=int(x), y=int(y))


def kv_inplace_default(device) -> bool:
    """Whether `TracedDecodeStepInPlace` should be the default on `device`.

    `COSYVOICE_KV_INPLACE` overrides either direction; this is only the fallback when
    it is unset. The trade is architecture-dependent -- in-place is worth 1.12-1.15x on
    Blackhole and 1.42x on Wormhole n300 (PERF.md, *Decode step, and what each change is
    worth*) -- so the default follows the architecture rather than picking one trade for
    both. `model.py` and `test_pipeline_perf.py` both call this rather than each
    hand-coding the check, so the test that is supposed to measure "whatever the model
    would actually run" cannot silently drift from what the model runs.
    """
    return "WORMHOLE" in str(device.arch())


class TtTransformerLayer:
    """`x = x + attn(norm1(x))`, `x = x + ff(norm2(x))`, with an optional KV cache."""

    def __init__(self, device, bag, meta, dtype=ttnn.bfloat16, cc=None, weights_dtype=None):
        self.device, self.dtype = device, dtype
        self.cc = accurate_compute_config(device) if cc is None else cc
        self.eps = meta["layer_norm_eps"]
        self.ffn_act = ttnn.relu if meta.get("ffn_activation", "relu") == "relu" else ttnn.silu
        self.attn = TtRelPosAttention(
            device, bag.sub("self_attn"), meta["n_head"], meta["d_k"], dtype, self.cc, weights_dtype
        )
        self.w1, self.b1 = _linear(device, bag, "feed_forward.w_1", dtype, weights_dtype)
        self.w2, self.b2 = _linear(device, bag, "feed_forward.w_2", dtype, weights_dtype)
        self.g1, self.bt1 = _layernorm_weights(device, bag, "norm1", dtype)
        self.g2, self.bt2 = _layernorm_weights(device, bag, "norm2", dtype)
        # `w_2` is `[d_ff, d_model]`, the largest and slowest op in a decode step, and it
        # is bound by its `K = d_ff` reduction rather than by weight traffic -- `w_1` holds
        # the same number of weight bytes and responds to `bfloat8_b` weights by -37 %
        # where this one responds by -2 %. Handing it a *small* explicit grid is what moves
        # it: at one row, spreading a 4096-deep reduction over the whole grid leaves each
        # core a sliver and the gather dominates. Measured standalone, `[1,1,4096] x
        # [4096,1024]`, bf16: 8x2 is 1.50x on n300 and 1.98x on p150b against the default,
        # and the default is indistinguishable from asking for the full grid explicitly.
        #
        # **Decode only.** The optimum is a property of `M = 1`. Prefill runs this same
        # linear at `M = 209`, where there is real work to spread and a 16-core grid would
        # be a pessimisation, so `__call__` applies it only when `T == 1`.
        self.ff2_grid = _core_grid_from_env("COSYVOICE_FF2_GRID")

    def __call__(
        self,
        x,
        pos_emb,
        mask=None,
        cache=None,
        return_cache=True,
        cache_free=False,
        bd_offset=None,
        cache_write=None,
    ):
        h = ttnn.layer_norm(x, weight=self.g1, bias=self.bt1, epsilon=self.eps)
        a, new_cache = self.attn.forward_cached(
            h,
            pos_emb,
            mask=mask,
            cache=cache,
            return_cache=return_cache,
            cache_free=cache_free,
            bd_offset=bd_offset,
            cache_write=cache_write,
        )
        ttnn.deallocate(h)
        x1 = ttnn.add(x, a)
        ttnn.deallocate(a)
        ttnn.deallocate(x)

        h = ttnn.layer_norm(x1, weight=self.g2, bias=self.bt2, epsilon=self.eps)
        f = ttnn.linear(h, self.w1, bias=self.b1, compute_kernel_config=self.cc)
        ttnn.deallocate(h)
        f = self.ffn_act(f)  # "relu" for this stack -- the text encoder uses SiLU
        ff2_kw = {"core_grid": self.ff2_grid} if (self.ff2_grid and f.shape[-2] == 1) else {}
        f2 = ttnn.linear(f, self.w2, bias=self.b2, compute_kernel_config=self.cc, **ff2_kw)
        ttnn.deallocate(f)
        out = ttnn.add(x1, f2)
        ttnn.deallocate(f2)
        ttnn.deallocate(x1)
        return out, new_cache


class TtARDecoder:
    """`TransformerEncoder.forward_chunk` -- prefill and one-token decode alike."""

    def __init__(self, device, bag, meta, dtype=ttnn.bfloat16, weights_dtype=None):
        self.device, self.dtype, self.meta = device, dtype, meta
        self.cc = accurate_compute_config(device)
        self.d_model = meta["d_model"]
        self.xscale = float(self.d_model) ** 0.5
        self.has_relu = bool(meta.get("embed_has_relu", False))
        self.embed_eps = meta.get("embed_norm_eps", 1e-5)

        self.w_in, self.b_in = _linear(device, bag, "embed.out.0", dtype)
        self.g_in, self.bt_in = _layernorm_weights(device, bag, "embed.out.1", dtype)
        self.layers = [
            TtTransformerLayer(device, bag.sub(f"encoders.{i}"), meta, dtype, self.cc, weights_dtype)
            for i in range(meta["n_layers"])
        ]
        self.g_after, self.bt_after = _layernorm_weights(device, bag, "after_norm", dtype)
        self._pos_cache: dict[int, object] = {}

    def enable_pos_proj_cache(self):
        """Let every layer cache `linear_pos(pos_emb)`, head-split and transposed.

        Safe only on the fixed-width paths, which is why it is a call rather than a
        constructor default: they pass the *same* `positional(max_len)` tensor on
        every step, so each layer's cache holds one entry for the whole utterance and
        never evicts. `forward_chunk`'s growing cache asks for a new width per token
        and must not turn this on.

        Worth a call of its own because that projection is the single largest matmul
        in the layer -- 511 rows through `[d_model, d_model]` against one row for each
        of q, k and v -- and it does not depend on the token being decoded.
        """
        for layer in self.layers:
            layer.attn.cache_pos_proj = True

    def release_pos_proj_cache(self):
        """Drop every layer's cached positional projection -- see
        `TtRelPosAttention.release_pos_proj_cache` for when this is safe."""
        for layer in self.layers:
            layer.attn.release_pos_proj_cache()

    def positional(self, key_size: int):
        """The `[1, 2*key_size - 1, d_model]` window, cached per size.

        Decoding grows the key size by one per token, so each step needs a window
        one element wider than the last. Generating it on the host and uploading is
        cheap next to 14 blocks of attention, but caching keeps the repeated
        prefill/verification passes from rebuilding it.
        """
        if key_size not in self._pos_cache:
            pos = espnet_rel_positional_encoding(key_size, self.d_model)
            self._pos_cache[key_size] = ttnn.from_torch(
                pos, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device
            )
        return self._pos_cache[key_size]

    def embed(self, xs):
        """Linear -> LayerNorm -> [ReLU] -> scale by sqrt(d_model)."""
        h = ttnn.linear(xs, self.w_in, bias=self.b_in, compute_kernel_config=self.cc)
        n = ttnn.layer_norm(h, weight=self.g_in, bias=self.bt_in, epsilon=self.embed_eps)
        ttnn.deallocate(h)
        if self.has_relu:
            r = ttnn.relu(n)
            ttnn.deallocate(n)
            n = r
        out = ttnn.multiply(n, self.xscale)
        ttnn.deallocate(n)
        return out

    def forward_chunk(self, xs, caches=None, mask=None):
        """xs `[1, chunk, input_size]`, `caches` a list of `(k, v)` per layer.

        Returns `(ys, new_caches)`. The old caches are consumed -- their k/v are
        concatenated into the new ones and then freed, so the caller must not hold
        a reference past this call.
        """
        chunk = xs.shape[1]
        cache_t1 = 0 if not caches else caches[0][0].shape[2]
        pos = self.positional(cache_t1 + chunk)

        h = self.embed(xs)
        new_caches = []
        for i, layer in enumerate(self.layers):
            cache = caches[i] if caches else None
            h, new_cache = layer(h, pos, mask=mask, cache=cache, return_cache=True)
            if cache is not None:
                for t in cache:
                    ttnn.deallocate(t)
            new_caches.append(new_cache)
        out = ttnn.layer_norm(h, weight=self.g_after, bias=self.bt_after, epsilon=self.meta["layer_norm_eps"])
        ttnn.deallocate(h)
        return out, new_caches

    @staticmethod
    def free_caches(caches):
        for cache in caches or []:
            for t in cache:
                ttnn.deallocate(t)

    # ----------------------------------------------------------------------
    # fixed-shape decoding
    # ----------------------------------------------------------------------
    def forward_chunk_fixed(self, xs, caches, max_len: int, valid: int, mask=None):
        """`forward_chunk` with a **right-aligned, fixed-width** KV cache.

        This is the difference between 0.4 and 35 tok/s, and the reason is not
        arithmetic. A cache that grows by one slot per token gives every step a new
        attention key size -- 210, 211, 212 -- and TTNN's program cache is keyed on
        shape, so *every token pays a fresh JIT compile*. Measured on Blackhole:
        28 ms for the first step, 3.3 s by the 32nd, and a second pass over the
        same sizes ran at 28 ms flat. **98.9% of the cold cost was compilation.**

        Holding the key width at `max_len` makes exactly two shapes exist, one for
        prefill and one for decode, no matter how long the utterance runs.

        The alignment is what makes it correct. ESPnet's `rel_shift` skews a
        `[t1, K]` score block on the assumption that the queries are the **last**
        `t1` of the `K` key positions -- that is precisely the streaming case it
        was written for. So the live tokens must sit at the *end* of the buffer and
        the padding at the front, not the other way round. Left-aligning instead
        gives every query the relative geometry of a position it is not at, which
        is wrong everywhere and obviously wrong nowhere.

        `valid` is how many of the `max_len` slots hold real history; the caller
        supplies `mask` suppressing the rest.
        """
        chunk = xs.shape[1]
        self.enable_pos_proj_cache()  # one window for the whole utterance -- see the method
        pos = self.positional(max_len)
        h = self.embed(xs)
        # One conversion for all 14 layers -- see `decode_mask`. Only a one-token step
        # can take the fused path; a prefill chunk has a real `[chunk, W]` skew.
        mask, mask_owned = self._fused_mask(mask, chunk)
        new_caches = []
        for i, layer in enumerate(self.layers):
            ck, cv = caches[i]
            # The attention concatenates `chunk` new slots onto whatever it is
            # given, so what it is given must be `max_len - chunk` wide. A cache
            # returned by a previous call is `max_len` wide and needs its oldest
            # `chunk` slots dropped; one straight from `empty_cache` is already
            # sized and must be left alone.
            b, nh, width, dk = ck.shape
            trimmed = width > max_len - chunk
            if trimmed:
                ck = ttnn.slice(ck, [0, 0, chunk, 0], [b, nh, max_len, dk])
                cv = ttnn.slice(cv, [0, 0, chunk, 0], [b, nh, max_len, dk])
                for t in caches[i]:
                    ttnn.deallocate(t)
            h, new_cache = layer(h, pos, mask=mask, cache=(ck, cv), return_cache=True)
            ttnn.deallocate(ck)
            ttnn.deallocate(cv)
            new_caches.append(new_cache)
        out = ttnn.layer_norm(h, weight=self.g_after, bias=self.bt_after, epsilon=self.meta["layer_norm_eps"])
        ttnn.deallocate(h)
        if mask_owned:
            ttnn.deallocate(mask)
        return out, new_caches

    # ------------------------------------------------------------------
    @property
    def sdpa_decode(self) -> bool:
        """Whether the layers will take the fused decode-attention path."""
        return bool(self.layers) and self.layers[0].attn.sdpa_decode

    def _fused_mask(self, mask, chunk: int):
        """`(mask, we_allocated_it)` — the per-head form when the fused path applies.

        Returned as a pair rather than assigned in place because the caller must not
        free a mask it was handed. Everything downstream keys on the mask's shape, so
        a `None` return here simply leaves the explicit chain in charge.
        """
        if mask is None or chunk != 1 or not self.sdpa_decode or mask.shape[-2] != 1:
            return mask, False
        return decode_mask(mask, self.meta["n_head"]), True

    def empty_cache(self, max_len: int, chunk: int, batch: int = 1):
        """Zeroed `[1, h, max_len - chunk, d_k]` k/v per layer.

        Sized so the attention's own concat with this step's `chunk` tokens lands
        on `max_len` -- the first chunk takes the same path as every later one, so
        there is no separate prefill shape to compile.
        """
        n_head, d_k = self.meta["n_head"], self.meta["d_k"]
        shape = (batch, n_head, max_len - chunk, d_k)
        return [
            (
                ttnn.zeros(shape, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device),
                ttnn.zeros(shape, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device),
            )
            for _ in range(self.meta["n_layers"])
        ]


class TracedDecodeStep:
    """One decode step captured as a trace and replayed with a single host command.

    Profiling measured the AR decoder at ~124 us per op over ~280 ops -- **dispatch
    bound, not compute bound**. Trace capture is the direct answer: it records the
    op graph once and replays it without re-issuing every op from the host.

    A trace replays *fixed device addresses*, which is what makes this more than a
    flag. Three things follow:

    * **The KV cache must live in persistent buffers.** The untraced path
      reallocates it with `concat` on every step, so a replay would write to
      addresses from the capture. Here the buffers are allocated once and updated
      with `ttnn.copy` at the end of the traced body -- read at the top, written at
      the bottom, which is safe within a single replay.
    * **Inputs must be preallocated too**, and refreshed with
      `copy_host_to_device_tensor` rather than a fresh `from_torch`. The token
      embedding changes every step, and the validity mask's *values* change even
      though its shape does not.
    * **Two warm-up passes precede capture**, so every kernel variant is
      JIT-compiled before the graph is recorded -- otherwise the compile lands
      inside the trace.

    Right-alignment is unchanged from `forward_chunk_fixed`: the live tokens sit at
    the end of the buffer because `rel_shift` assumes the queries are the last of
    the key positions.
    """

    def __init__(self, decoder, max_len: int, batch: int = 1):
        self.dec = decoder
        self.max_len = max_len
        # One decode step for `batch` sequences at once. Every buffer below simply
        # gains a leading dimension: the AR decoder's own ops are written against
        # `[B, ...]` already, and `right_aligned_bias` takes one `valid` per row, so
        # sequences with different prompt lengths batch with no further bookkeeping.
        # `TtTransformerLM.generate_batch` is the caller, and its docstring carries
        # why it is worth doing -- a decode step at one row is bound by reading the
        # decoder's weights out of DRAM, and that read is shared across the batch.
        #
        # Only the moving cache is batched. `TracedDecodeStepInPlace` captures 65
        # traces where this captures one, and multiplying that by a batch is a trace
        # region no board here has. The two are within 1.15x of each other on
        # Blackhole, and batching moves far more than 1.15x.
        self.batch = batch
        # Before the first `_body()`, so the projection is computed and cached during
        # warm-up and the trace records only the read.
        decoder.enable_pos_proj_cache()
        meta = decoder.meta
        self.h, self.d_k, self.n_layers = meta["n_head"], meta["d_k"], meta["n_layers"]
        d_in = meta["input_size"]
        dev, dt = decoder.device, decoder.dtype

        self.x_buf = ttnn.from_torch(torch.zeros(batch, 1, d_in), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        # Heads on dim 2 when the fused path is on, built on the **host**. Doing the
        # expansion with a device `ttnn.repeat` inside the traced body is what took
        # traced-vs-untraced from 1.0 to 0.918: the op is correct on its own
        # (`scripts/probe_sdpa_decode.py` controls for it) but not as the per-step input to
        # a replayed trace. The mask is
        # rebuilt on the host every step anyway, so emitting it already expanded costs
        # nothing and removes the op from the trace entirely.
        self.mask_heads = self.h if decoder.sdpa_decode else 1
        self.mask_buf = ttnn.from_torch(
            right_aligned_bias(max_len, [max_len] * batch, 1, heads=self.mask_heads),
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
        )
        # Time-major `[1, T, h, d_k]`, not `[1, h, T, d_k]`. `TILE_LAYOUT` tiles the last
        # two dims, so this puts the time axis on a *free* one -- appending a token then
        # costs 19.7 us instead of 207.2, and a 13.9 us permute puts it back in
        # `[1, h, T, d_k]` for the matmuls. See `TtRelPosAttention.forward_cached`.
        shape = (batch, max_len, self.h, self.d_k)
        self.k_buf = [
            ttnn.from_torch(torch.zeros(shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            for _ in range(self.n_layers)
        ]
        self.v_buf = [
            ttnn.from_torch(torch.zeros(shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            for _ in range(self.n_layers)
        ]
        # Persistent output buffer. The tensor a traced body *returns* lives in the
        # trace's own pool, and reading it after a replay is fragile -- copying into
        # a buffer allocated up front is the same discipline the KV cache uses, for
        # the same reason.
        self.ys = ttnn.from_torch(torch.zeros(batch, 1, meta["d_model"]), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        self.trace_id = None

    # ------------------------------------------------------------------
    def seed(self, caches):
        """Load a prefill's caches into the persistent buffers.

        The prefill runs untraced -- it happens once per utterance and its shape
        differs from the decode step's, so tracing it would buy a single dispatch
        saving for a second compile.
        """
        for i, (k, v) in enumerate(caches):
            # The prefill runs the ordinary path and hands back `[1, h, T, d_k]`; the
            # buffers are time-major, so each is permuted once here. Once per utterance,
            # against 164 appends that the layout makes cheap.
            kf = ttnn.permute(k, (0, 2, 1, 3))
            vf = ttnn.permute(v, (0, 2, 1, 3))
            ttnn.copy(kf, self.k_buf[i])
            ttnn.copy(vf, self.v_buf[i])
            ttnn.deallocate(kf)
            ttnn.deallocate(vf)

    def _body(self):
        """The graph that gets traced. Reads the buffers, writes them back."""
        pos = self.dec.positional(self.max_len)
        h = self.dec.embed(self.x_buf)
        for i, layer in enumerate(self.dec.layers):
            # Drop the oldest slot so the attention's own concat lands on max_len.
            trimmed = (
                ttnn.slice(self.k_buf[i], [0, 1, 0, 0], [self.batch, self.max_len, self.h, self.d_k]),
                ttnn.slice(self.v_buf[i], [0, 1, 0, 0], [self.batch, self.max_len, self.h, self.d_k]),
            )
            h, (k_new, v_new) = layer(h, pos, mask=self.mask_buf, cache=trimmed, return_cache=True, cache_free=True)
            for t in trimmed:
                ttnn.deallocate(t)
            ttnn.copy(k_new, self.k_buf[i])
            ttnn.copy(v_new, self.v_buf[i])
            ttnn.deallocate(k_new)
            ttnn.deallocate(v_new)
        out = ttnn.layer_norm(
            h, weight=self.dec.g_after, bias=self.dec.bt_after, epsilon=self.dec.meta["layer_norm_eps"]
        )
        ttnn.deallocate(h)
        ttnn.copy(out, self.ys)
        ttnn.deallocate(out)

    def capture(self):
        """Warm up twice, then record. Always closes the trace, even on failure --
        an open trace wedges the device."""
        for _ in range(2):
            self._body()
        ttnn.synchronize_device(self.dec.device)

        self.trace_id = ttnn.begin_trace_capture(self.dec.device, cq_id=0)
        try:
            self._body()
        finally:
            ttnn.end_trace_capture(self.dec.device, self.trace_id, cq_id=0)
        return self

    def step(self, x_host: torch.Tensor, valid):
        """One token per row in, `[B, 1, d]` hidden states out. Buffers updated in place.

        `valid` is one int at batch 1, or one int per row when batching -- the rows
        genuinely differ, because batched utterances start from prompts of different
        lengths.
        """
        valids = (
            [min(valid, self.max_len)] * self.batch if isinstance(valid, int) else [min(v, self.max_len) for v in valid]
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(x_host, dtype=self.dec.dtype, layout=ttnn.TILE_LAYOUT), self.x_buf
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                right_aligned_bias(self.max_len, valids, 1, heads=self.mask_heads),
                dtype=self.dec.dtype,
                layout=ttnn.TILE_LAYOUT,
            ),
            self.mask_buf,
        )
        ttnn.execute_trace(self.dec.device, self.trace_id, cq_id=0, blocking=False)
        return self.ys

    def release(self):
        if self.trace_id is not None:
            ttnn.release_trace(self.dec.device, self.trace_id)
            self.trace_id = None
        for t in [self.x_buf, self.mask_buf, self.ys, *self.k_buf, *self.v_buf]:
            ttnn.deallocate(t)


class TracedDecodeStepInPlace:
    """The same decode step, with a KV cache that is written rather than rebuilt.

    `TracedDecodeStep` keeps the newest token at the last row of a `max_len` buffer,
    which means every step must **move the whole cache down by one**. Time-major
    storage made that move cheap -- a free-axis append is 19.7 us where a tiled one
    is 207 -- but cheap is not free, and what is left is still the largest block in
    the step: per tensor per layer, a slice, two permutes, a concat and a writeback
    copy, ~95 us together, 14 layers and two tensors deep.

    The way out is to stop moving it. Hold a buffer `TILE` rows wider than the
    window, write token `i` of the current group straight into row `max_len + i`
    with `ttnn.update_cache` (3.7 us, in place), and let the buffer walk forward
    through its scratch zone for 32 steps. Only then does anything move, and what
    moves is a **32-row, tile-aligned** shift -- the one shift `TILE_LAYOUT` can do
    without re-tiling. Amortised, 95 us a tensor a layer a step becomes about 11.

    **The scratch zone is two tiles, not one, and that is not a tuning choice.** A
    decode step's cost tracks the *parity* of its key-axis tile count, not its size:
    swept on Blackhole at a 384-row window, 10/12/14/16 tiles cost 6.32/6.73/7.09/
    7.99 ms while 11/13/15 cost 7.33/7.92/8.54 -- every odd count about a millisecond
    dearer than its even neighbours, which is `out_subblock_w` falling back to 1. A
    one-tile scratch zone turns a 12-tile window into a 13-tile buffer and pays that
    penalty on every step: measured, +1.21 ms, against the +0.82 ms the in-place write
    is worth. The mechanism was never the problem; the odd width was. Two tiles keeps
    the parity, doubles the interval between shifts, and costs two traces per row
    instead of one.

    Two more things follow, and both are the reason this is a separate class
    rather than a flag:

    * **The query is no longer the last key position**, so `rel_shift`'s `T = 1`
      identity is false and the positional window has to be selected explicitly.
      `bd_offset = scratch - 1 - slot` does it; the derivation is in `forward_cached`.
    * **Both `update_idx` and that offset are baked at capture**, because a trace
      records runtime arguments, not just addresses. So there is one trace per scratch
      row, plus one for the periodic shift. (`paged_update_cache` takes its index as a
      *device tensor* and would need only one trace -- but it wants a paged block
      cache, not this layout, and rejects it.)

    Those 65 traces are the practical cost of the design, and the trace region has to
    be sized for them. How much is not a tidy multiple: offered 64 MB the capture
    asked for 68.6, offered 128 MB it asked for 134.3, so the allocator fills what it
    is given before reporting a shortfall and neither number is "the requirement".
    384 MB is a size this has been observed to capture in. A device opened with the
    usual 64 MB fails at capture -- which is why this is opt-in rather than default,
    and why `generate()` catches that failure and falls back rather than dying.

    The mask does the rest of the bookkeeping. Exactly `max_len` rows are live at
    every sub-step -- `[slot + 1, max_len + slot]` -- so the rows the shift discards
    are precisely the rows the mask has already been suppressing, and the window
    slides at the same rate it does in the moving version.
    """

    TILE = 32  # `TILE_LAYOUT`'s row granularity: the one shift width that is free
    SCRATCH_TILES = 2  # keeps the buffer's tile count the same parity as the window

    def __init__(self, decoder, max_len: int, scratch_tiles: int = SCRATCH_TILES):
        self.dec = decoder
        self.max_len = max_len
        self.scratch = self.TILE * scratch_tiles
        self.width = max_len + self.scratch
        decoder.enable_pos_proj_cache()
        meta = decoder.meta
        self.h, self.d_k, self.n_layers = meta["n_head"], meta["d_k"], meta["n_layers"]
        # See `TracedDecodeStep.__init__`: the head expansion is a host concern, and
        # it matters more here — this class captures 65 traces, not one.
        self.mask_heads = self.h if decoder.sdpa_decode else 1
        dev, dt = decoder.device, decoder.dtype

        self.x_buf = ttnn.from_torch(
            torch.zeros(1, 1, meta["input_size"]), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev
        )
        self.mask_buf = ttnn.from_torch(
            slot_bias(self.width, self.scratch, max_len, 0, heads=self.mask_heads),
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
        )
        shape = (1, self.h, self.width, self.d_k)
        self.k_buf = [
            ttnn.from_torch(torch.zeros(shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            for _ in range(self.n_layers)
        ]
        self.v_buf = [
            ttnn.from_torch(torch.zeros(shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            for _ in range(self.n_layers)
        ]
        # The scratch zone the shift re-zeroes, allocated once. It is *read* inside a
        # trace, which is fine; building it with `ttnn.zeros` in there would not be,
        # since that is a host->device write.
        self.pad = ttnn.from_torch(
            torch.zeros(1, self.h, self.scratch, self.d_k), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev
        )
        self.ys = ttnn.from_torch(torch.zeros(1, 1, meta["d_model"]), dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        self.traces = [None] * self.scratch
        self.shift_trace = None
        self.slot = 0

    # ------------------------------------------------------------------
    def seed(self, caches):
        """Load a prefill's `[1, h, max_len, d_k]` caches into rows `[0, max_len)`.

        The scratch tail is zeroed rather than left as whatever the warm-up passes
        wrote. Nothing reads it -- row `max_len + slot` is always written before it
        is attended to, and the mask covers the rest -- but a masking mistake should
        surface as a reproducible wrong answer, not as luck with old data.
        """
        for i, (k, v) in enumerate(caches):
            for src, dst in ((k, self.k_buf[i]), (v, self.v_buf[i])):
                wide = ttnn.concat([src, self.pad], dim=2)
                ttnn.copy(wide, dst)
                ttnn.deallocate(wide)
        self.slot = 0

    def _body(self, slot: int):
        """The graph traced for one sub-step. `slot` is baked into it."""
        pos = self.dec.positional(self.width)
        h = self.dec.embed(self.x_buf)
        for i, layer in enumerate(self.dec.layers):
            h, _ = layer(
                h,
                pos,
                mask=self.mask_buf,
                cache=(self.k_buf[i], self.v_buf[i]),
                return_cache=False,
                cache_write=self.max_len + slot,
                bd_offset=(self.scratch - 1) - slot,
            )
        out = ttnn.layer_norm(
            h, weight=self.dec.g_after, bias=self.dec.bt_after, epsilon=self.dec.meta["layer_norm_eps"]
        )
        ttnn.deallocate(h)
        ttnn.copy(out, self.ys)
        ttnn.deallocate(out)

    def _shift_body(self):
        """Slide every buffer down by `TILE` rows and re-zero the tail.

        The slice starts at row 32, so it is tile-aligned and is a genuine copy
        rather than the alias a full-extent slice would give -- which is what makes
        writing the result back over its own source safe.
        """
        for i in range(self.n_layers):
            for buf in (self.k_buf[i], self.v_buf[i]):
                keep = ttnn.slice(buf, [0, 0, self.scratch, 0], [1, self.h, self.width, self.d_k])
                wide = ttnn.concat([keep, self.pad], dim=2)
                ttnn.deallocate(keep)
                ttnn.copy(wide, buf)
                ttnn.deallocate(wide)

    def capture(self):
        """Two warm-up passes, then `TILE + 1` traces.

        The warm-up is per slot, not once overall. All 32 sub-steps do hit the same
        program-cache entries -- `update_cache` hashes its op type and tensors, not
        its index -- but "should hit" is a poor thing to rely on inside a capture,
        where a compile is an error rather than a slow path.
        """
        for _ in range(2):
            self._body(0)
        self._shift_body()
        ttnn.synchronize_device(self.dec.device)

        for slot in range(self.scratch):
            self._body(slot)  # warm this slot's runtime-arg variant
            tid = ttnn.begin_trace_capture(self.dec.device, cq_id=0)
            try:
                self._body(slot)
            finally:
                ttnn.end_trace_capture(self.dec.device, tid, cq_id=0)
            self.traces[slot] = tid

        self.shift_trace = ttnn.begin_trace_capture(self.dec.device, cq_id=0)
        try:
            self._shift_body()
        finally:
            ttnn.end_trace_capture(self.dec.device, self.shift_trace, cq_id=0)
        return self

    def step(self, x_host: torch.Tensor, valid: int):
        """One token in, `[1, 1, d]` hidden state out.

        The shift is enqueued behind the step it completes rather than ahead of the
        next one. It touches only the KV buffers, so the `ys` this returns is already
        final; the caller's synchronise covers both.
        """
        slot = self.slot
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(x_host, dtype=self.dec.dtype, layout=ttnn.TILE_LAYOUT), self.x_buf
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                slot_bias(self.width, self.scratch, valid, slot, heads=self.mask_heads),
                dtype=self.dec.dtype,
                layout=ttnn.TILE_LAYOUT,
            ),
            self.mask_buf,
        )
        ttnn.execute_trace(self.dec.device, self.traces[slot], cq_id=0, blocking=False)

        self.slot = slot + 1
        if self.slot == self.scratch:
            ttnn.execute_trace(self.dec.device, self.shift_trace, cq_id=0, blocking=False)
            self.slot = 0
        return self.ys

    def release(self):
        for tid in [*self.traces, self.shift_trace]:
            if tid is not None:
                ttnn.release_trace(self.dec.device, tid)
        self.traces = [None] * self.scratch
        self.shift_trace = None
        for t in [self.x_buf, self.mask_buf, self.pad, self.ys, *self.k_buf, *self.v_buf]:
            ttnn.deallocate(t)


def slot_bias(width: int, tile: int, valid: int, slot: int, heads: int = 1, dtype=torch.float32):
    """Additive `[1, 1, 1, width]` mask for an in-place cache at sub-step `slot`.

    The token just written sits at row `max_len + slot`, and the window is the
    `min(valid, max_len)` rows ending there. Everything else is suppressed: rows
    below have aged out of the window, rows above are scratch not yet written.

    Because the live span always ends at `max_len + slot` and is at most `max_len`
    long, it starts at row `slot + 1` once the window is full -- so after `tile`
    sub-steps the rows that have aged out are exactly rows `[0, tile)`, which is
    what `_shift_body` discards. The mask and the shift are two views of one window.
    """
    max_len = width - tile
    hi = max_len + slot
    lo = hi - min(valid, max_len) + 1
    rows = torch.arange(width)
    live = (rows >= lo) & (rows <= hi)
    m = torch.where(live, 0.0, NEG_INF).reshape(1, 1, 1, width)
    if heads > 1:  # the per-head form sdpa_decode wants -- see `decode_mask`
        m = m.expand(1, 1, heads, width).contiguous()
    return m.to(dtype)


def right_aligned_bias(max_len: int, valid, chunk: int = 1, causal: bool = False, heads: int = 1, dtype=torch.float32):
    """Additive `[B, 1, chunk, max_len]` mask for a right-aligned cache.

    Slots before `max_len - valid` are padding and are suppressed. When `causal`,
    query `i` (sitting at slot `max_len - chunk + i`) additionally may not see any
    slot beyond its own.

    **`valid` may be a sequence, one entry per batch row**, which is what makes a
    batched decode step possible at all. Utterances batched together have different
    prompt lengths and stop at different tokens, so at any given step they have
    different amounts of real history -- but because the cache is *right*-aligned,
    each row's live span always ends at the last slot and differs only in where it
    starts. One mask row per sequence expresses exactly that, and nothing else about
    the decode step has to know that the rows are unequal. A left-aligned cache would
    need per-row gather instead.
    """
    valids = [valid] if isinstance(valid, int) else list(valid)
    slots = torch.arange(max_len)
    live = torch.stack([slots >= (max_len - v) for v in valids])  # [B, max_len]
    m = torch.where(live, 0.0, NEG_INF).reshape(len(valids), 1, 1, max_len).repeat(1, 1, chunk, 1)
    if causal:
        q_slot = (max_len - chunk) + torch.arange(chunk)
        m = torch.where(slots.reshape(1, 1, 1, -1) <= q_slot.reshape(1, 1, -1, 1), m, NEG_INF)
    if heads > 1:  # decode only: chunk is 1, so dim 2 is free to carry heads instead
        m = m.expand(len(valids), 1, heads, max_len).contiguous()
    return m.to(dtype)
