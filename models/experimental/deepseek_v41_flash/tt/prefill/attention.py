# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash prefill attention, on top of the V4-Flash prefill block.

V4.1 keeps V4's prefill block -- q_a -> q_b, shared-KV MQA, the attention sink, inverse RoPE on the output, the
grouped o_a -> o_b and the chunk state
(:class:`~models.experimental.deepseek_v4_flash.tt.prefill.attention.PrefillAttentionState`) -- and replaces its
compressed path, as the decode block does (see :mod:`..decode.attention`):

* ``compress_ratios[i]`` is 0 (window only), 2 (softmax-gated pooling of 2 tokens, no overlap) or 1 (one
  latent per token). Only ``kv_source_layer_ids`` compress; later layers of a group read the latest source's
  compressed KV.
* The index key is ``k_norm(wk(latent))``, RoPE'd, from the source's pre-RoPE latent. V4's indexer
  compressor is gone.
* ``index_source_layer_ids`` score those keys and select the top ``index_topk`` entries per query; later
  layers reuse that selection.
* q_b's per-head RMSNorm is gone.

Attention builds no mask: every layer runs ``ttnn.transformer.deepseek_v4_prefill_sdpa`` over its sliding window
``[tail | chunk]`` and, for a compressed layer, the compressed entries its index source selected. The selection is
each query's top-``index_topk`` entry rows, from ``ttnn.transformer.deepseek_v4_prefill_indexer_select``, which
scores and selects in one pass over the entries each query can see (a query that sees fewer than ``index_topk``
fills the rest with a sentinel); the attention op skips sentinels and window rows before position 0.

Layers share state, so the layers of a chunk run in order and hand it down in a :class:`PrefillShared`:
the latest kv source's :class:`~models.experimental.deepseek_v4_flash.tt.prefill.attention.PrefillAttentionState`
(compressed KV and index keys), the latest index source's selected rows and the candidates.

V4.1's selection is two-level (the reference's ``select_candidate_blocks``). The ``candidate_source_layer_id``
(layer 20) scores each block of ``candidate_block_size`` entries by its best index score, always keeps the block
holding the query's newest entry, and keeps the best ``candidate_topk_blocks`` blocks per query
(``deepseek_v4_prefill_indexer_select_candidates``): the candidates, handed down in :class:`PrefillShared` as
block ids. Every index source from the candidate source on (20, 24, 28, 32, 36) then selects its
top-``index_topk`` among its candidate blocks' entries only (``deepseek_v4_prefill_indexer_select_from_candidates``),
so its cost follows the ``candidate_topk_blocks * candidate_block_size`` pool rather than the context, as decode's
does. The source's own top-``index_topk`` lies inside its candidates, so restricting it changes nothing; while a
query sees no more than ``candidate_topk_blocks`` blocks every one is a candidate, so the select is single-level.

Traced prefill (:meth:`DeepSeekV41PrefillAttention.forward_static`) runs V4's static attention over persistent
buffers: a kv source writes its entries and index keys into front-anchored buffers at the packet's rows
(``deepseek_v4_prefill_write_rows``), and the layers of a stage hand them (and the selection) down in a
:class:`PrefillStaticShared`. The chunk's start reaches every op as a device tensor and every op reads only the
rows the chunk can see, so one trace serves every chunk of a prompt at a cost that follows its position.
"""

import dataclasses
from typing import Optional

import torch
import ttnn

from models.experimental.deepseek_v4_flash.tt.layers import Linear
from models.experimental.deepseek_v4_flash.tt.prefill.attention import (
    _SDPA_CHUNK,
    ALIGNMENT,
    SLIDING_ATTENTION,
    DeepSeekV4PrefillAttention,
    PrefillAttentionState,
    PrefillStaticBuffers,
)
from models.experimental.deepseek_v4_flash.tt.weight_cache import _as_cache


@dataclasses.dataclass
class PrefillShared:
    """What the layers of one prompt hand down to the layers after them.

    ``kv_source`` is the latest kv source's state: the compressed KV (``compressed_kv``) and index keys
    (``idx_keys``) the layers of its group read. ``entry_idx`` is the latest index source's selection for the
    current chunk: uint32 ROW_MAJOR ``[1, 1, T, index_topk]``, each query's selected entry rows (a query that sees
    fewer than ``index_topk`` entries fills the rest with ``0xFFFFFFFF``, which the attention op skips). Each
    chunk's index source replaces it, so run a chunk's layers in order.

    ``candidates`` is the candidate source's level-one selection for the current chunk: uint32 ROW_MAJOR
    ``[1, 1, T, candidate_topk_blocks]``, each query's candidate block ids (unused slots ``0xFFFFFFFF``), which the
    index sources from the candidate source on select among.
    """

    kv_source: Optional[PrefillAttentionState] = None
    entry_idx: Optional[ttnn.Tensor] = None
    candidates: Optional[ttnn.Tensor] = None


@dataclasses.dataclass
class PrefillStaticShared:
    """:class:`PrefillShared` for traced prefill: what the layers of one stage hand down within one chunk.

    ``entries`` / ``idx_keys`` are the current group's persistent front-anchored buffers on this stage (the kv source's
    own, or the stage's copy of a source on an earlier stage), ``entry_idx`` and ``candidates`` as in
    :class:`PrefillShared`. A kv source with ``keep_rows`` set also leaves the rows it wrote this chunk in
    ``new_rows`` (``(entries, keys)``, ROW_MAJOR ``[1, 1, T / rate, width]``), for the next stage.
    """

    entries: Optional[ttnn.Tensor] = None
    idx_keys: Optional[ttnn.Tensor] = None
    entry_idx: Optional[ttnn.Tensor] = None
    keep_rows: bool = False
    new_rows: Optional[tuple] = None
    candidates: Optional[ttnn.Tensor] = None


@dataclasses.dataclass
class V41PrefillStaticStep:
    """The per-chunk inputs of a traced V4.1 attention layer, computed inside the trace from the chunk's packet.

    As V4's :class:`~models.experimental.deepseek_v4_flash.tt.prefill.attention.PrefillStaticStep`, without masks:
    ``rope[kind]`` = ``(cos, sin)`` ``[1, 1, T, rope_dim]`` for the chunk's tokens, ``entry_rope[rate]`` the same for
    the chunk's new entries, ``entry_rows[rate]`` (uint32 ``[T / rate]``) the rows the chunk's entries go to.

    ``start`` is the chunk's start position, int32 ROW_MAJOR ``[1, 1, 1, 1]``, which the indexer and attention ops
    read on device. ``fresh`` marks a bounded replay's first chunk: its window starts at the chunk (the tail holds
    no token).
    """

    rope: dict
    entry_rope: dict
    start: ttnn.Tensor
    fresh: bool = False
    entry_rows: Optional[dict] = None


def ratio_layer_type(rate: int) -> str:
    """The layer type of a compressed layer of ``rate`` (one per compress rate; V4.1's ratios share a config type)."""
    return f"compressed_ratio_{rate}"


class DeepSeekV41PrefillAttention(DeepSeekV4PrefillAttention):
    """V4.1 prefill attention for layer ``layer_idx``.

    ``weights`` comes from :func:`~..config.attention_weights`; ``rope`` is V4's bundle, of which a layer reads
    ``rope["main"]`` (window only) or ``rope["compress"]`` (every compressed layer). The compressor and indexer
    are replicated over a TP mesh, so every rank holds the same state, as in decode. Weights the checkpoint
    keeps in bf16 (compressor, index ``wk`` / ``weights_proj``) stay bf16; the rest use ``weight_dtype``.
    """

    q_head_norm = False

    def compress_rate(self, config) -> Optional[int]:
        return config.compress_ratios[self.layer_idx] or None

    def __init__(
        self,
        config,
        layer_idx: int,
        weights: dict,
        device,
        rope: dict,
        cache=None,
        weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        tp_size: int = 1,
    ):
        super().__init__(
            config,
            layer_idx,
            weights,
            device,
            rope,
            cache=cache,
            weight_dtype=weight_dtype,
            tp_size=tp_size,
        )
        cache = _as_cache(cache)
        self.layer_type = ratio_layer_type(self.rate) if self.rate else SLIDING_ATTENTION
        if self.rate not in (None, 1, 2):
            raise ValueError(f"layer {layer_idx}: no compressor for ratio {self.rate}")
        # bf16 ROW_MAJOR [1, 1, 1, H_local]: the sinks (pre-divided by the scale) in the attention op's layout.
        self.prefill_sinks = ttnn.reshape(
            ttnn.to_layout(self.sinks, ttnn.ROW_MAJOR_LAYOUT), [1, 1, 1, self.local_num_heads]
        )
        self.is_kv_source = layer_idx in config.kv_source_layer_ids
        self.is_index_source = layer_idx in config.index_source_layer_ids
        # A source's new entries only reach the layers after it through its own selection.
        assert self.is_index_source or not self.is_kv_source
        assert self.rate or not (self.is_kv_source or self.is_index_source)
        source = config.candidate_source_layer_id
        self.is_candidate_source = self.is_index_source and layer_idx == source
        self.uses_candidates = self.is_index_source and 0 <= source < layer_idx

        def gamma(name: str) -> ttnn.Tensor:
            weight = weights[name]
            return self._to_device((weight() if callable(weight) else weight).detach().float().reshape(1, 1, 1, -1))

        if self.is_kv_source:
            self.c_kv_proj = Linear(
                weights["compressor.wkv.weight"],
                device,
                cache.file("compressor.wkv.prefill"),
                dtype=ttnn.bfloat16,
                mesh_mapper=self._replicate,
            )
            if self.rate == 2:
                self.c_gate_proj = Linear(
                    weights["compressor.wgate.weight"],
                    device,
                    cache.file("compressor.wgate.prefill"),
                    dtype=ttnn.bfloat16,
                    mesh_mapper=self._replicate,
                )
            self.c_norm_weight = gamma("compressor.norm.weight")
            self.i_k_proj = Linear(
                weights["indexer.wk.weight"],
                device,
                cache.file("indexer.wk.prefill"),
                dtype=ttnn.bfloat16,
                mesh_mapper=self._replicate,
            )
            self.i_k_norm_weight = gamma("indexer.k_norm.weight")
        if self.is_index_source:
            # Replicated on every rank, as in V4: the selection is a head-sum, so replicating avoids a collective.
            self.index_heads = config.index_n_heads
            self.index_head_dim = config.index_head_dim
            if self.index_head_dim < self.rope_dim or self.index_head_dim % ttnn.TILE_SIZE:
                raise ValueError(f"index_head_dim {self.index_head_dim} unsupported (rope_dim {self.rope_dim})")
            # The fused select's widths: deepseek_v4_prefill_indexer_select returns 512 rows, the candidate mode
            # 2048 blocks of 8.
            select = (config.index_topk, config.candidate_topk_blocks, config.candidate_block_size)
            if select != (512, 2048, 8):
                raise ValueError(f"layer {layer_idx}: the fused indexer select needs (512, 2048, 8), got {select}")
            self.i_q_b_proj = Linear(
                weights["indexer.wq_b.weight"],
                device,
                cache.file("indexer.wq_b.prefill"),
                dtype=weight_dtype,
                mesh_mapper=self._replicate,
            )
            self.i_weights_proj = Linear(
                weights["indexer.weights_proj.weight"],
                device,
                cache.file("indexer.weights_proj.prefill"),
                dtype=ttnn.bfloat16,
                mesh_mapper=self._replicate,
            )
            # relu(q.k) * softmax_scale, weighted by weights_proj(x) * n_heads^-0.5, summed over heads.
            self.index_scale = float(self.index_head_dim**-0.5 * self.index_heads**-0.5)

    def _compress(self, hidden: ttnn.Tensor, state: PrefillAttentionState) -> ttnn.Tensor:
        """Pool this chunk's windows into compressed entries and index keys, append both to ``state``, and return
        every entry so far (ROW_MAJOR). Only a kv source calls it.

        The chunk is window-aligned, so no partial window crosses a chunk boundary. Entry ``w`` is RoPE'd at the
        position of its window's first token, ``state.seq_len + w * rate``.
        """
        cos, sin = self._rope_tables(state.seq_len + torch.arange(hidden.shape[2] // self.rate) * self.rate)
        entries, keys = self._pool_entries(hidden, cos, sin)
        if state.compressed_kv is not None:
            entries = ttnn.concat([state.compressed_kv, entries], dim=2)
            keys = ttnn.concat([state.idx_keys, keys], dim=2)
        state.compressed_kv, state.idx_keys = entries, keys
        return entries

    def _pool_entries(self, hidden: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """This chunk's ``T / rate`` compressed entries and index keys, ROW_MAJOR ``[1, 1, T / rate, Dh]`` and
        ``[1, 1, T / rate, index_head_dim]``, RoPE'd with ``(cos, sin)`` (the entries' window starts)."""
        n_win = hidden.shape[2] // self.rate
        dh = self.head_dim
        pooled = self.c_kv_proj(hidden)
        if self.rate == 2:
            gate = self.c_gate_proj(hidden)
            # Each window's two tokens side by side, in fp32: slot 0 is columns [0, Dh), slot 1 [Dh, 2*Dh).
            kv, gate = (self._window_view(x, n_win, 2 * dh) for x in (pooled, gate))
            k0, k1, g0, g1 = (
                ttnn.slice(x, [0, 0, 0, j * dh], [1, 1, n_win, (j + 1) * dh]) for x in (kv, gate) for j in (0, 1)
            )
            # The softmax over a pair is sigmoid(g0 - g1); a softmax over the 2-row window axis would run into
            # its tile padding.
            w0 = ttnn.sigmoid(ttnn.subtract(g0, g1))
            pooled = ttnn.typecast(ttnn.add(k1, ttnn.multiply(w0, ttnn.subtract(k0, k1))), ttnn.bfloat16)
        latent = ttnn.rms_norm(pooled, weight=self.c_norm_weight, epsilon=self.eps)  # pre-RoPE
        entries = ttnn.to_layout(self._rope(latent, cos, sin), ttnn.ROW_MAJOR_LAYOUT)
        keys = ttnn.rms_norm(self.i_k_proj(latent), weight=self.i_k_norm_weight, epsilon=self.eps)
        keys = ttnn.to_layout(self._rope(keys, cos, sin), ttnn.ROW_MAJOR_LAYOUT)
        return entries, keys

    # ------------------------------------------------------------------ #
    # selection
    # ------------------------------------------------------------------ #
    def _index_inputs(
        self, latent: ttnn.Tensor, hidden: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """The fused select's ``(q, weights)``: the post-RoPE indexer queries ``[1, Hi, T, Di]`` and the per-head
        gates with every scale folded in ``[1, 1, T, Hi]``, both bf16 TILE in DRAM."""
        flat = self.i_q_b_proj(latent)
        heads, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            flat,
            num_heads=self.index_heads,
            num_kv_heads=0,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(flat)
        q = self._rope(heads, cos, sin)
        ttnn.deallocate(heads)
        gates = self.i_weights_proj(hidden)
        weights = ttnn.multiply(gates, self.index_scale)
        ttnn.deallocate(gates)
        return q, weights

    def _select(
        self,
        latent: ttnn.Tensor,
        hidden: ttnn.Tensor,
        idx_keys: ttnn.Tensor,
        start: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        shared,
    ) -> None:
        """Replace ``shared.entry_idx`` (a :class:`PrefillShared` or :class:`PrefillStaticShared`) with this index
        source's selection for the chunk; the candidate source also replaces ``shared.candidates``.

        ``idx_keys`` is the group's ROW_MAJOR ``[1, 1, E, Di]`` index keys (only the rows a query can see are read)
        and ``start`` the chunk's start, a 1-element device tensor."""
        q, weights = self._index_inputs(latent, hidden, cos, sin)
        if shared.entry_idx is not None:
            ttnn.deallocate(shared.entry_idx)
            shared.entry_idx = None
        if self.is_candidate_source:
            if shared.candidates is not None:
                ttnn.deallocate(shared.candidates)
            shared.candidates = ttnn.transformer.deepseek_v4_prefill_indexer_select_candidates(
                q, weights, idx_keys, start, compress_ratio=self.rate
            )
        if self.is_candidate_source or self.uses_candidates:
            if shared.candidates is None:
                raise ValueError(f"layer {self.layer_idx} runs before the candidate source has seen this chunk")
            shared.entry_idx = ttnn.transformer.deepseek_v4_prefill_indexer_select_from_candidates(
                q, weights, idx_keys, start, shared.candidates, compress_ratio=self.rate
            )
        else:
            shared.entry_idx = ttnn.transformer.deepseek_v4_prefill_indexer_select(
                q, weights, idx_keys, start, compress_ratio=self.rate
            )
        ttnn.deallocate(q)
        ttnn.deallocate(weights)

    def forward(
        self,
        hidden: ttnn.Tensor,
        state: Optional[PrefillAttentionState] = None,
        shared: Optional[PrefillShared] = None,
    ) -> ttnn.Tensor:
        """Attention over one prompt chunk.

        ``hidden`` is ``[1, 1, T, D]`` bf16 TILE on the device; ``state`` is this layer's own
        :class:`~models.experimental.deepseek_v4_flash.tt.prefill.attention.PrefillAttentionState` (``None`` runs a
        whole prompt from scratch), advanced in place, and ``shared`` the :class:`PrefillShared` of the prompt
        (``None``: a fresh one, which only a layer that is its own kv source can use). The layers of a chunk
        must run in layer order. Returns ``[1, 1, T, D]`` bf16 TILE.

        Under TP ``hidden`` must be replicated on every rank, and the result is replicated too.
        """
        if state is None:
            state = self.new_state()
        if shared is None:
            shared = PrefillShared()
        self._check_chunk(hidden, state)
        num_tokens = hidden.shape[2]
        start = state.seq_len

        cos, sin = self._rope_tables(start + torch.arange(num_tokens))
        q, latent = self._q_stem(hidden, cos, sin, return_latent=True)
        kv = self._kv_stem(hidden, cos, sin)
        position = ttnn.from_torch(
            torch.tensor([[[[start]]]], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._replicate,
        )

        entries = None
        if self.rate:
            if self.is_kv_source:
                entries = self._compress(hidden, state)
                shared.kv_source = state
            else:
                source = shared.kv_source
                if source is None or source.num_entries != (start + num_tokens) // self.rate:
                    raise ValueError(f"layer {self.layer_idx} runs before its kv source has seen this chunk")
                entries = source.compressed_kv
            if self.is_index_source:
                self._select(latent, hidden, shared.kv_source.idx_keys, position, cos, sin, shared)
            elif shared.entry_idx is None:
                raise ValueError(f"layer {self.layer_idx} runs before its index source has seen this chunk")

        sw = self.sliding_window
        chunk_rm = ttnn.to_layout(kv, ttnn.ROW_MAJOR_LAYOUT)
        tail = state.kv_tail if state.kv_tail is not None else self._zeros_rm(sw)
        window = ttnn.concat([tail, chunk_rm], dim=2)  # [1, 1, sw + T, Dh]
        ttnn.deallocate(chunk_rm)
        state.kv_tail = ttnn.slice(window, [0, 0, num_tokens, 0], [1, 1, num_tokens + sw, self.head_dim])
        attn = self._attend_sdpa(q, window, position, cos, sin, entries, shared.entry_idx if self.rate else None)
        ttnn.deallocate(window)
        ttnn.deallocate(position)
        out = self._o_proj(attn)
        state.seq_len += num_tokens
        return out

    # ------------------------------------------------------------------ #
    # traced prefill
    # ------------------------------------------------------------------ #
    def new_static_buffers(self, chunk_size: int, entry_capacity: int = 0) -> PrefillStaticBuffers:
        """Persistent zeroed state for :meth:`forward_static`: the K=V ``tail`` and last ``window`` of every layer,
        and a kv source's ``entry_capacity`` front-anchored entry and index-key rows. V4.1 has no pooling overlap.

        Allocated once, before any trace exists.
        """
        bufs = PrefillStaticBuffers(
            tail=self._zeros_rm(self.sliding_window), window=self._zeros_rm(self.sliding_window + chunk_size)
        )
        if self.is_kv_source:
            if entry_capacity <= 0 or entry_capacity % ALIGNMENT:
                raise ValueError(f"entry_capacity {entry_capacity} must be a positive multiple of {ALIGNMENT}")
            bufs.entries = self._zeros_rm(entry_capacity)
            bufs.idx_keys = self._zeros_rm(entry_capacity, self.config.index_head_dim)
        return bufs

    def reset_static(self, bufs: PrefillStaticBuffers) -> None:
        """Rewind ``bufs`` to the start of a prompt: zero the tail. The entry rows are front-anchored and a prompt
        reads only the ones it wrote (see V4's :meth:`~DeepSeekV4PrefillAttention.reset_static`)."""
        shape = tuple(bufs.tail.shape)
        if getattr(self, "_tail_zeros", (None, None))[0] != shape:
            self._tail_zeros = (shape, self.host_tensor(torch.zeros(shape), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT))
        ttnn.copy_host_to_device_tensor(self._tail_zeros[1], bufs.tail)

    def forward_static(
        self, hidden: ttnn.Tensor, bufs: PrefillStaticBuffers, step: V41PrefillStaticStep, shared: PrefillStaticShared
    ) -> ttnn.Tensor:
        """Trace-safe :meth:`forward`: no host traffic, no growing tensors.

        Positions come from ``step`` (RoPE rows, the entries' RoPE and rows, the start), this layer's state is
        ``bufs`` (updated in place) and the group's state ``shared``. ``hidden`` ``[1, 1, T, D]`` -> ``[1, 1, T, D]``.

        Every op reads only the rows the chunk can see: the kv source writes its ``T / rate`` new rows at the
        packet's rows, and the select and attention ops take the whole persistent buffers but read the visible
        prefix (or the candidates' rows) at the start they read on device.
        """
        cos, sin = step.rope[self.rope_kind]
        q, latent = self._q_stem(hidden, cos, sin, return_latent=True)
        kv = self._kv_stem(hidden, cos, sin)
        if self.rate:
            if self.is_kv_source:
                entries, keys = self._pool_entries(hidden, *step.entry_rope[self.rate])
                rows = step.entry_rows[self.rate]
                ttnn.transformer.deepseek_v4_prefill_write_rows(entries, bufs.entries, rows)
                ttnn.transformer.deepseek_v4_prefill_write_rows(keys, bufs.idx_keys, rows)
                if shared.keep_rows:
                    shared.new_rows = (entries, keys)
                else:
                    ttnn.deallocate(entries)
                    ttnn.deallocate(keys)
                shared.entries, shared.idx_keys = bufs.entries, bufs.idx_keys
            if self.is_index_source:
                self._select(latent, hidden, shared.idx_keys, step.start, cos, sin, shared)

        num_tokens = q.shape[2]
        sw = self.sliding_window
        chunk_rm = ttnn.to_layout(kv, ttnn.ROW_MAJOR_LAYOUT)
        window = ttnn.concat([bufs.tail, chunk_rm], dim=2)  # [1, 1, sw + T, Dh]
        ttnn.deallocate(chunk_rm)
        self._write_rows(window, bufs.window, 0)
        new_tail = ttnn.slice(window, [0, 0, num_tokens, 0], [1, 1, num_tokens + sw, self.head_dim])
        self._write_rows(new_tail, bufs.tail, 0)
        ttnn.deallocate(new_tail)
        attn = self._attend_sdpa(
            q,
            window,
            step.start,
            cos,
            sin,
            shared.entries if self.rate else None,
            shared.entry_idx if self.rate else None,
            empty_tail=step.fresh,
        )
        ttnn.deallocate(window)
        return self._o_proj(attn)

    def _attend_sdpa(
        self,
        q: ttnn.Tensor,
        window: ttnn.Tensor,
        start: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        entries: Optional[ttnn.Tensor] = None,
        rows: Optional[ttnn.Tensor] = None,
        empty_tail: bool = False,
    ) -> ttnn.Tensor:
        """``deepseek_v4_prefill_sdpa`` of ``q`` ``[1, H_local, T, Dh]`` over its sliding window in ``window`` (ROW_MAJOR
        ``[tail | chunk]``) and, for a compressed layer, the rows ``rows`` (uint32 ROW_MAJOR ``[1, 1, T, k]``) of the
        ROW_MAJOR ``entries``, then undo V's RoPE. ``start`` is the chunk's start, a 1-element device tensor. The op
        skips entry rows a query has not seen yet and window rows before position 0 (every tail row with
        ``empty_tail``). Returns ``[1, H_local, T, Dh]`` TILE."""
        query = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.transformer.deepseek_v4_prefill_sdpa(
            query,
            window,
            start,
            cache_kv=entries,
            cache_idx=rows,
            window_size=self.sliding_window,
            scale=self.scaling,
            k_chunk_size=_SDPA_CHUNK,
            attention_sink=self.prefill_sinks,
            compress_ratio=self.rate or 0,
            empty_tail=empty_tail,
        )
        ttnn.deallocate(query)
        attn = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        ttnn.deallocate(out)
        # K = V, so V carried K's RoPE on its trailing slice: rotate it back at the query's position.
        return self._rope(attn, cos, ttnn.neg(sin))
