# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash prefill attention, on top of the V4-Flash prefill block.

V4.1 keeps V4's prefill block -- q_a -> q_b, shared-KV MQA over ``[tail | chunk | compressed entries]``
as one dense masked SDPA, the attention sink, inverse RoPE on the output, the grouped o_a -> o_b, the
chunk state (:class:`~models.experimental.deepseek_v4_flash.tt.prefill.attention.PrefillAttentionState`)
and the selection folded into the additive mask -- and replaces its compressed path, as the decode block
does (see :mod:`..decode.attention`):

* ``compress_ratios[i]`` is 0 (window only), 2 (softmax-gated pooling of 2 tokens, no overlap) or 1 (one
  latent per token). Only ``kv_source_layer_ids`` compress; later layers of a group read the latest source's
  compressed KV.
* The index key is ``k_norm(wk(latent))``, RoPE'd, from the source's pre-RoPE latent. V4's indexer
  compressor is gone.
* ``index_source_layer_ids`` score those keys and select the top ``index_topk`` entries per query; later
  layers reuse that selection.
* q_b's per-head RMSNorm is gone.

Layers share state, so the layers of a chunk run in order and hand it down in a :class:`PrefillShared`:
the latest kv source's :class:`~models.experimental.deepseek_v4_flash.tt.prefill.attention.PrefillAttentionState`
(compressed KV and index keys) and the latest index source's selection mask.

V4.1's selection is two-level past ``candidate_topk_blocks * candidate_block_size`` compressed entries (the
reference's ``select_candidate_blocks``). The ``candidate_source_layer_id`` (layer 20) scores each block of
``candidate_block_size`` entries by its best index score, always keeps the block holding the query's newest entry,
and keeps the best ``candidate_topk_blocks`` blocks per query: the candidates, handed down in
:class:`PrefillShared` as an additive entry mask. The index sources after it (24, 28, 32, 36) restrict their own
scores to the candidates before their top-``index_topk``; the source's own top-``index_topk`` is unrestricted.
Below the threshold every reachable block is kept, so the selection is V4's single threshold top-k and no
candidates are built.

Traced prefill (:meth:`DeepSeekV41PrefillAttention.forward_static`) runs V4's static attention over persistent
buffers: a kv source writes its entries and index keys into front-anchored buffers at the packet's rows, and the
layers of a stage hand them (and the selection) down in a :class:`PrefillStaticShared`. Once every query of a
position tier sees ``index_topk`` entries and a whole window, the selection is the index source's ``topk`` rows and
attention is ``sparse_sdpa`` over each query's selected and window rows, with no mask; earlier tiers keep the dense
masked SDPA.
"""

import dataclasses
from typing import Optional

import torch
import ttnn

from models.experimental.deepseek_v4_flash.tt.common import _MASK_NEG
from models.experimental.deepseek_v4_flash.tt.layers import Linear
from models.experimental.deepseek_v4_flash.tt.prefill.attention import (
    _SDPA_CHUNK,
    ALIGNMENT,
    SLIDING_ATTENTION,
    DeepSeekV4PrefillAttention,
    PrefillAttentionState,
    PrefillStaticBuffers,
    PrefillStaticStep,
)
from models.experimental.deepseek_v4_flash.tt.weight_cache import _as_cache

# Entry columns per pass of the candidate blocks' max: 2 x 128 KB of L1 row staging, a multiple of the tile width.
_BLOCK_MAX_COLUMNS = 65536


@dataclasses.dataclass
class PrefillShared:
    """What the layers of one prompt hand down to the layers after them.

    ``kv_source`` is the latest kv source's state: the compressed KV (``compressed_kv``) and index keys
    (``idx_keys``) the layers of its group read. ``entry_sel`` is the latest index source's selection for the
    current chunk: an additive TILE mask ``[1, 1, T, E]`` (0 on each query's selected entries, -inf elsewhere),
    or ``None`` while there are no more than ``index_topk`` entries and every visible one is selected. Each
    chunk's index source replaces it, so run a chunk's layers in order.

    ``candidates`` is the candidate source's level-one selection for the current chunk, the additive TILE mask
    ``[1, 1, T, E]`` the later index sources restrict their scores to (0 on the visible entries of each query's
    kept blocks, -inf elsewhere), or ``None`` while there are no more than
    ``candidate_topk_blocks * candidate_block_size`` entries.
    """

    kv_source: Optional[PrefillAttentionState] = None
    entry_sel: Optional[ttnn.Tensor] = None
    candidates: Optional[ttnn.Tensor] = None


@dataclasses.dataclass
class PrefillStaticShared:
    """:class:`PrefillShared` for traced prefill: what the layers of one stage hand down within one chunk.

    ``entries`` / ``idx_keys`` are the current group's persistent front-anchored buffers on this stage (the kv source's
    own, or the stage's copy of a source on an earlier stage), ``entry_sel`` the latest index source's selection
    ``[1, 1, T, cap]`` as an additive mask, or, in a tier that attends sparsely (:attr:`V41PrefillStaticStep.sparse`),
    ``entry_idx`` its selected entry rows (uint32 ROW_MAJOR ``[1, 1, T, index_topk]``). A kv source with
    ``keep_rows`` set also leaves copies of the rows it wrote this chunk in ``new_rows`` (``(entries, keys)``,
    ROW_MAJOR ``[1, 1, T / rate, width]``), for the next stage.

    ``candidates`` is :attr:`PrefillShared.candidates` over the ``cap`` entry rows (built by the candidate source
    on this stage, or expanded from the block mask an earlier stage sent). A candidate source with
    ``keep_candidates`` set also leaves its compact block mask in ``candidate_blocks`` (0/1 bf16 ROW_MAJOR
    ``[1, 1, T, cap / candidate_block_size]``), for the next stage.
    """

    entries: Optional[ttnn.Tensor] = None
    idx_keys: Optional[ttnn.Tensor] = None
    entry_sel: Optional[ttnn.Tensor] = None
    entry_idx: Optional[ttnn.Tensor] = None
    keep_rows: bool = False
    new_rows: Optional[tuple] = None
    candidates: Optional[ttnn.Tensor] = None
    keep_candidates: bool = False
    candidate_blocks: Optional[ttnn.Tensor] = None


@dataclasses.dataclass
class V41PrefillStaticStep(PrefillStaticStep):
    """V4's :class:`PrefillStaticStep` plus the candidate source's pins.

    ``candidate_pins[layer_type]`` is bf16 TILE ``[1, 1, T, cap / candidate_block_size]``: ``-_MASK_NEG`` on the
    block holding each query's newest entry, 0 elsewhere (made from :func:`candidate_pin_keys` and the chunk's
    start). Present only on the candidate source's stage, in position tiers whose ``cap`` holds more blocks than
    ``candidate_topk_blocks``.

    ``sparse`` holds the compressed layer types that attend through ``sparse_sdpa`` in this tier: those whose every
    query sees at least ``index_topk`` entries and a whole window, so no selected or window row needs masking.
    ``window_idx[cap]`` is uint32 ROW_MAJOR ``[1, 1, T, sliding_window]``: query ``i``'s window rows
    ``cap + i + 1 .. cap + i + sliding_window`` of the keys ``[cap entry rows | tail | chunk]``. ``q_pad`` is the
    bf16 ROW_MAJOR ``[1, pad, T, Dh]`` zero heads that round the local query heads up to a tile for the op.
    """

    candidate_pins: Optional[dict] = None
    sparse: frozenset = frozenset()
    window_idx: Optional[dict] = None
    q_pad: Optional[ttnn.Tensor] = None


def ratio_layer_type(rate: int) -> str:
    """The key a compressed layer of ``rate`` uses in a :class:`PrefillStaticStep` (masks, caps, index pads)."""
    return f"compressed_ratio_{rate}"


def candidate_pin_keys(num_tokens: int, num_blocks: int, rate: int, block_size: int) -> torch.Tensor:
    """fp32 ``[T, num_blocks]``: query ``i`` of a chunk starting at ``start`` pins block ``b`` iff ``key[i, b] == start``.

    The pinned block holds the query's newest entry, ``(compress_len - 1) // block_size`` with
    ``compress_len = (start + i + 1) // rate``. For ``start`` a multiple of ``rate * block_size`` that is
    ``start / (rate * block_size) + ((i + 1) // rate - 1) // block_size``, so the key is
    ``rate * block_size * (b - ((i + 1) // rate - 1) // block_size)``. A query with no entry yet (``start == 0``,
    ``i + 1 < rate``) pins nothing: its keys are all positive. The keys are integers far below ``2**24``, so an fp32
    compare on device is exact.
    """
    query = torch.arange(num_tokens).view(-1, 1)
    block = torch.arange(num_blocks).view(1, -1)
    return (rate * block_size * (block - ((query + 1) // rate - 1) // block_size)).float()


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
        maskless: bool = False,
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
            maskless=maskless,
        )
        cache = _as_cache(cache)
        # V4's static step keys masks and entry caps by layer type; V4.1's two ratios share one config type.
        self.layer_type = ratio_layer_type(self.rate) if self.rate else SLIDING_ATTENTION
        if self.rate not in (None, 1, 2):
            raise ValueError(f"layer {layer_idx}: no compressor for ratio {self.rate}")
        self.sparse_sinks = None  # bf16 ROW_MAJOR [1, 1, 1, H_local padded to a tile], sparse_sdpa's sink layout
        if self.rate:
            local = self.local_num_heads
            sinks = ttnn.reshape(ttnn.to_layout(self.sinks, ttnn.ROW_MAJOR_LAYOUT), [1, 1, 1, local])
            pad = -local % ttnn.TILE_SIZE
            if pad:
                zeros = ttnn.from_torch(
                    torch.zeros(1, 1, 1, pad),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=device,
                    mesh_mapper=self._replicate,
                )
                sinks = ttnn.concat([sinks, zeros], dim=3)
            self.sparse_sinks = sinks
        self.is_kv_source = layer_idx in config.kv_source_layer_ids
        self.is_index_source = layer_idx in config.index_source_layer_ids
        # A source's new entries only reach the layers after it through its own selection.
        assert self.is_index_source or not self.is_kv_source
        assert self.rate or not (self.is_kv_source or self.is_index_source)
        self.max_entries = config.candidate_topk_blocks * config.candidate_block_size
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        source = config.candidate_source_layer_id
        self.is_candidate_source = self.is_index_source and layer_idx == source
        self.uses_candidates = self.is_index_source and 0 <= source < layer_idx
        if self.is_candidate_source and ALIGNMENT % (self.rate * self.candidate_block_size):
            raise ValueError(
                f"layer {layer_idx}: chunk starts (multiples of {ALIGNMENT}) must be whole candidate blocks of "
                f"{self.rate * self.candidate_block_size} tokens"
            )

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
    # two-level select
    # ------------------------------------------------------------------ #
    def _select_from_scores(self, scores: ttnn.Tensor, cut: ttnn.Tensor) -> ttnn.Tensor:
        """The top-``index_topk`` selection mask ``[1, 1, T, E]`` from indexer ``scores`` (``cut`` already added), as
        V4's :meth:`~DeepSeekV4PrefillAttention._select_entries` thresholds them. ``cut`` is added back, so entries the
        threshold keeps only because a query has fewer than ``index_topk`` reachable ones are dropped again."""
        k = min(self.index_topk, scores.shape[3])
        theta = ttnn.min(ttnn.topk(scores, k=k, dim=-1, largest=True, sorted=True)[0], dim=-1, keepdim=True)
        return ttnn.add(ttnn.log(ttnn.ge(scores, theta)), cut)

    def _candidate_blocks(self, scores: ttnn.Tensor, pins: ttnn.Tensor) -> ttnn.Tensor:
        """Level one of the two-level select: the 0/1 bf16 TILE mask ``[1, 1, T, E / block_size]`` of each query's
        ``candidate_topk_blocks`` best blocks.

        ``scores`` ``[1, 1, T, E]`` TILE has the causal cut added; a block scores its best entry. ``pins`` (shaped
        like the result) lifts the block holding each query's newest entry above every other, so it is always kept.
        Every block scoring at least the k-th best is kept: a query with fewer than ``k`` reachable blocks keeps
        unreachable ones too, which the causal cut drops when the mask is expanded (:meth:`_expand_candidates`).
        """
        t, e = scores.shape[2], scores.shape[3]
        size = self.candidate_block_size
        if e % size:
            raise ValueError(f"layer {self.layer_idx}: {e} entries are not whole blocks of {size}")
        # A row-major strided slice per position in the block: a max over a [.., size] axis would read tile padding.
        # The strided slice stages two whole input rows per core in L1, so it runs over column segments.
        maxima = []
        for lo in range(0, e, _BLOCK_MAX_COLUMNS):
            hi = min(e, lo + _BLOCK_MAX_COLUMNS)
            segment = scores if hi - lo == e else ttnn.slice(scores, [0, 0, 0, lo], [1, 1, t, hi])
            rows = ttnn.to_layout(segment, ttnn.ROW_MAJOR_LAYOUT)
            if segment is not scores:
                ttnn.deallocate(segment)
            best = None
            for j in range(size):
                part = ttnn.slice(rows, [0, 0, 0, j], [1, 1, t, hi - lo], [1, 1, 1, size])
                tiled = ttnn.to_layout(part, ttnn.TILE_LAYOUT)
                ttnn.deallocate(part)
                if best is None:
                    best = tiled
                    continue
                merged = ttnn.maximum(best, tiled)
                ttnn.deallocate(best)
                ttnn.deallocate(tiled)
                best = merged
            ttnn.deallocate(rows)
            maxima.append(best)
        if len(maxima) == 1:
            best = maxima[0]
        else:
            best = ttnn.concat(maxima, dim=3)
            for piece in maxima:
                ttnn.deallocate(piece)
        pinned = ttnn.add(best, pins)
        ttnn.deallocate(best)
        k = min(self.candidate_topk_blocks, e // size)
        theta = ttnn.min(ttnn.topk(pinned, k=k, dim=-1, largest=True, sorted=True)[0], dim=-1, keepdim=True)
        blocks = ttnn.ge(pinned, theta)
        ttnn.deallocate(pinned)
        ttnn.deallocate(theta)
        return blocks

    def _expand_candidates(self, blocks: ttnn.Tensor, cut: ttnn.Tensor) -> ttnn.Tensor:
        """The block mask ``blocks`` (:meth:`_candidate_blocks`, TILE or ROW_MAJOR) as the additive TILE entry mask
        ``[1, 1, T, E]``: ``cut`` (the causal cut over the entries) on each query's kept blocks, -inf elsewhere."""
        entries = ttnn.repeat_interleave(blocks, self.candidate_block_size, dim=3)
        if entries.layout != ttnn.TILE_LAYOUT:
            tiled = ttnn.to_layout(entries, ttnn.TILE_LAYOUT)
            ttnn.deallocate(entries)
            entries = tiled
        additive = ttnn.log(entries)
        ttnn.deallocate(entries)
        mask = ttnn.add(additive, cut)
        ttnn.deallocate(additive)
        return mask

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
                if shared.entry_sel is not None:
                    ttnn.deallocate(shared.entry_sel)
                shared.entry_sel = None
                if self.is_candidate_source and shared.candidates is not None:
                    ttnn.deallocate(shared.candidates)
                    shared.candidates = None
                num_entries = entries.shape[2]
                # Up to index_topk entries top-k selects every visible one (the plain causal set); past it, select.
                if num_entries > self.index_topk:
                    keys = dataclasses.replace(state, idx_keys=shared.kv_source.idx_keys)
                    scores, cut = self._index_scores(latent, hidden, keys, cos, sin)
                    if self.is_candidate_source and num_entries > self.max_entries:
                        num_blocks = num_entries // self.candidate_block_size
                        pin_keys = candidate_pin_keys(num_tokens, num_blocks, self.rate, self.candidate_block_size)
                        pins = self._to_device(
                            torch.where(pin_keys == start, -_MASK_NEG, 0.0).reshape(1, 1, num_tokens, num_blocks)
                        )
                        blocks = self._candidate_blocks(scores, pins)
                        ttnn.deallocate(pins)
                        shared.candidates = self._expand_candidates(blocks, cut)
                        ttnn.deallocate(blocks)
                    elif self.uses_candidates and num_entries > self.max_entries:
                        if shared.candidates is None or shared.candidates.shape[3] != num_entries:
                            raise ValueError(
                                f"layer {self.layer_idx} runs before the candidate source has seen this chunk"
                            )
                        restricted = ttnn.add(scores, shared.candidates)
                        ttnn.deallocate(scores)
                        scores, cut = restricted, shared.candidates
                    shared.entry_sel = self._select_from_scores(scores, cut)
                    ttnn.deallocate(scores)

        if self.maskless:
            attn = self._attend_maskless(q, kv, state, cos, sin)
        else:
            attn = self._attend(q, kv, entries, state, cos, sin, shared.entry_sel)
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

        Positions come from ``step`` (RoPE rows, the entries' RoPE and rows, masks and the tier's entry rows ``cap``),
        this layer's state is ``bufs`` (updated in place) and the group's state ``shared``. ``hidden`` ``[1, 1, T, D]``
        -> ``[1, 1, T, D]``.

        The candidate source builds candidates when ``step`` carries its pins (the tier's ``cap`` holds more than
        ``candidate_topk_blocks`` blocks); a later index source restricts its scores to ``shared.candidates`` when set.
        While the real entries fit ``candidate_topk_blocks`` blocks every reachable block is kept, so a tier that
        builds candidates still selects as the plain top-k does on its early chunks.

        In a tier where the layer type is in ``step.sparse``, an index source keeps its selected rows
        (``shared.entry_idx``) instead of a mask and the layer attends through :meth:`_attend_sparse`.
        """
        cos, sin = step.rope[self.rope_kind]
        q, latent = self._q_stem(hidden, cos, sin, return_latent=True)
        kv = self._kv_stem(hidden, cos, sin)
        sparse = self.layer_type in step.sparse
        if self.rate:
            if self.is_kv_source:
                entries, keys = self._pool_entries(hidden, *step.entry_rope[self.rate])
                if shared.keep_rows:
                    shared.new_rows = (ttnn.clone(entries), ttnn.clone(keys))
                rows, cap = step.entry_rows[self.rate], step.caps[self.layer_type]
                self._write_entries(bufs.entries, entries, rows, cap)
                self._write_entries(bufs.idx_keys, keys, rows, cap)
                shared.entries, shared.idx_keys = bufs.entries, bufs.idx_keys
            if self.is_index_source:
                for old in (shared.entry_sel, shared.entry_idx):
                    if old is not None:
                        ttnn.deallocate(old)
                shared.entry_sel = shared.entry_idx = None
                keys = dataclasses.replace(bufs, idx_keys=shared.idx_keys)
                scores, cut = self._index_scores_static(latent, hidden, keys, step, cos, sin)
                pins = (step.candidate_pins or {}).get(self.layer_type) if self.is_candidate_source else None
                if pins is not None:
                    blocks = self._candidate_blocks(scores, pins)
                    if shared.candidates is not None:
                        ttnn.deallocate(shared.candidates)
                    shared.candidates = self._expand_candidates(blocks, cut)
                    if shared.keep_candidates:
                        shared.candidate_blocks = ttnn.to_layout(blocks, ttnn.ROW_MAJOR_LAYOUT)
                    ttnn.deallocate(blocks)
                elif self.uses_candidates and shared.candidates is not None:
                    restricted = ttnn.add(scores, shared.candidates)
                    ttnn.deallocate(scores)
                    scores, cut = restricted, shared.candidates
                if sparse:
                    shared.entry_idx = self._select_rows(scores)
                else:
                    shared.entry_sel = self._select_from_scores(scores, cut)
                ttnn.deallocate(scores)
        if sparse:
            attn = self._attend_sparse(q, kv, shared.entries, bufs, step, cos, sin, shared.entry_idx)
        else:
            attn = self._attend_static(
                q,
                kv,
                shared.entries if self.rate else None,
                bufs,
                step,
                cos,
                sin,
                shared.entry_sel if self.rate else None,
            )
        return self._o_proj(attn)

    def _select_rows(self, scores: ttnn.Tensor) -> ttnn.Tensor:
        """The top-``index_topk`` entry rows of each query, uint32 ROW_MAJOR ``[1, 1, T, index_topk]``, from indexer
        ``scores`` (causal cut added). In a sparse tier every query sees at least ``index_topk`` entries, so every
        row picked is a visible one. Unlike :meth:`_select_from_scores`, ties at the threshold keep exactly
        ``index_topk`` rows."""
        values, rows = ttnn.topk(scores, k=self.index_topk, dim=-1, largest=True, sorted=True)
        ttnn.deallocate(values)
        if rows.dtype != ttnn.uint32:
            wide = ttnn.typecast(rows, ttnn.uint32)
            ttnn.deallocate(rows)
            rows = wide
        out = ttnn.to_layout(rows, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(rows)
        return out

    def _attend_sparse(
        self,
        q: ttnn.Tensor,
        kv: ttnn.Tensor,
        entries: ttnn.Tensor,
        bufs: PrefillStaticBuffers,
        step: V41PrefillStaticStep,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        rows: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """:meth:`_attend_static` with no mask: ``sparse_sdpa`` gathers each query's ``rows`` (its selected entries)
        and its window rows (``step.window_idx``) out of the keys ``[the tier's cap entry rows | tail | chunk]``.

        The tail rolls in place as in :meth:`_attend_static`. The op takes a tile of query heads, so the local heads
        are padded with ``step.q_pad`` and the padding sliced off the output.
        """
        num_tokens = q.shape[2]
        sw = self.sliding_window
        cap = step.caps[self.layer_type]
        chunk_rm = ttnn.to_layout(kv, ttnn.ROW_MAJOR_LAYOUT)
        window = ttnn.concat([bufs.tail, chunk_rm], dim=2)  # [1, 1, sw + T, Dh]
        ttnn.deallocate(chunk_rm)
        self._write_rows(window, bufs.window, 0)
        new_tail = ttnn.slice(window, [0, 0, num_tokens, 0], [1, 1, num_tokens + sw, self.head_dim])
        self._write_rows(new_tail, bufs.tail, 0)
        ttnn.deallocate(new_tail)

        head, owned = self._entry_prefix(entries, cap)
        keys = ttnn.concat([head, window], dim=2)
        ttnn.deallocate(window)
        if owned:
            ttnn.deallocate(head)
        indices = ttnn.concat([rows, step.window_idx[cap]], dim=3)  # [1, 1, T, index_topk + sw]
        query = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)
        if step.q_pad is not None:
            padded = ttnn.concat([query, step.q_pad], dim=1)
            ttnn.deallocate(query)
            query = padded
        out = ttnn.transformer.sparse_sdpa(
            query,
            keys,
            indices,
            self.head_dim,
            kv_format=ttnn.transformer.SparseKVFormat.BF16,
            scale=self.scaling,
            k_chunk_size=_SDPA_CHUNK,
            attention_sink=self.sparse_sinks,
        )
        ttnn.deallocate(query)
        ttnn.deallocate(keys)
        ttnn.deallocate(indices)
        if out.shape[1] != self.local_num_heads:
            local = ttnn.slice(out, [0, 0, 0, 0], [1, self.local_num_heads, num_tokens, self.head_dim])
            ttnn.deallocate(out)
            out = local
        attn = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        ttnn.deallocate(out)
        return self._rope(attn, cos, ttnn.neg(sin))
