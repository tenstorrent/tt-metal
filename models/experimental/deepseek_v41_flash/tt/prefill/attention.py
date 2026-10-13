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

Both levels of V4.1's two-level selection reduce to V4's single threshold top-k while the compressed
entries fit in the candidate pre-filter (``candidate_topk_blocks * candidate_block_size`` entries, the
bound the decode caches assert too): the pre-filter then keeps every reachable block. Past it
:meth:`DeepSeekV41PrefillAttention._check_chunk` refuses the chunk.

Traced prefill (:meth:`DeepSeekV41PrefillAttention.forward_static`) runs V4's static attention over persistent
buffers: a kv source writes its entries and index keys into front-anchored buffers at the packet's rows, and the
layers of a stage hand them (and the selection) down in a :class:`PrefillStaticShared`.
"""

import dataclasses
from typing import Optional

import torch
import ttnn

from models.experimental.deepseek_v4_flash.tt.layers import Linear
from models.experimental.deepseek_v4_flash.tt.prefill.attention import (
    ALIGNMENT,
    SLIDING_ATTENTION,
    DeepSeekV4PrefillAttention,
    PrefillAttentionState,
    PrefillStaticBuffers,
    PrefillStaticStep,
)
from models.experimental.deepseek_v4_flash.tt.weight_cache import _as_cache


@dataclasses.dataclass
class PrefillShared:
    """What the layers of one prompt hand down to the layers after them.

    ``kv_source`` is the latest kv source's state: the compressed KV (``compressed_kv``) and index keys
    (``idx_keys``) the layers of its group read. ``entry_sel`` is the latest index source's selection for the
    current chunk: an additive TILE mask ``[1, 1, T, E]`` (0 on each query's selected entries, -inf elsewhere),
    or ``None`` while there are no more than ``index_topk`` entries and every visible one is selected. Each
    chunk's index source replaces it, so run a chunk's layers in order.
    """

    kv_source: Optional[PrefillAttentionState] = None
    entry_sel: Optional[ttnn.Tensor] = None


@dataclasses.dataclass
class PrefillStaticShared:
    """:class:`PrefillShared` for traced prefill: what the layers of one stage hand down within one chunk.

    ``entries`` / ``idx_keys`` are the current group's persistent front-anchored buffers on this stage (the kv source's
    own, or the stage's copy of a source on an earlier stage), ``entry_sel`` the latest index source's selection
    ``[1, 1, T, cap]``. A kv source with ``keep_rows`` set also leaves copies of the rows it wrote this chunk in
    ``new_rows`` (``(entries, keys)``, ROW_MAJOR ``[1, 1, T / rate, width]``), for the next stage.
    """

    entries: Optional[ttnn.Tensor] = None
    idx_keys: Optional[ttnn.Tensor] = None
    entry_sel: Optional[ttnn.Tensor] = None
    keep_rows: bool = False
    new_rows: Optional[tuple] = None


def ratio_layer_type(rate: int) -> str:
    """The key a compressed layer of ``rate`` uses in a :class:`PrefillStaticStep` (masks, caps, index pads)."""
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
        self.is_kv_source = layer_idx in config.kv_source_layer_ids
        self.is_index_source = layer_idx in config.index_source_layer_ids
        # A source's new entries only reach the layers after it through its own selection.
        assert self.is_index_source or not self.is_kv_source
        assert self.rate or not (self.is_kv_source or self.is_index_source)
        self.max_entries = config.candidate_topk_blocks * config.candidate_block_size

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

    def _check_chunk(self, hidden: ttnn.Tensor, state: PrefillAttentionState) -> None:
        super()._check_chunk(hidden, state)
        if self.rate and (state.seq_len + hidden.shape[2]) // self.rate > self.max_entries:
            raise NotImplementedError(
                f"layer {self.layer_idx} would hold more than {self.max_entries} compressed entries after this "
                "chunk, past which V4.1's candidate pre-filter drops blocks: it is not implemented in prefill"
            )

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
                # Up to index_topk entries top-k selects every visible one (the plain causal set); past it, select.
                shared.entry_sel = None
                if entries.shape[2] > self.index_topk:
                    keys = dataclasses.replace(state, idx_keys=shared.kv_source.idx_keys)
                    shared.entry_sel = self._select_entries(latent, hidden, keys, cos, sin)

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
            if entry_capacity > self.max_entries:
                raise NotImplementedError(
                    f"layer {self.layer_idx}: {entry_capacity} entry rows exceed the {self.max_entries} the candidate "
                    "pre-filter keeps whole; it is not implemented in prefill"
                )
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
        self, hidden: ttnn.Tensor, bufs: PrefillStaticBuffers, step: PrefillStaticStep, shared: PrefillStaticShared
    ) -> ttnn.Tensor:
        """Trace-safe :meth:`forward`: no host traffic, no growing tensors.

        Positions come from ``step`` (RoPE rows, the entries' RoPE and rows, masks and the tier's entry rows ``cap``),
        this layer's state is ``bufs`` (updated in place) and the group's state ``shared``. ``hidden`` ``[1, 1, T, D]``
        -> ``[1, 1, T, D]``.
        """
        cos, sin = step.rope[self.rope_kind]
        q, latent = self._q_stem(hidden, cos, sin, return_latent=True)
        kv = self._kv_stem(hidden, cos, sin)
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
                if shared.entry_sel is not None:
                    ttnn.deallocate(shared.entry_sel)
                keys = dataclasses.replace(bufs, idx_keys=shared.idx_keys)
                shared.entry_sel = self._select_static(latent, hidden, keys, step, cos, sin)
        attn = self._attend_static(
            q, kv, shared.entries if self.rate else None, bufs, step, cos, sin, shared.entry_sel if self.rate else None
        )
        return self._o_proj(attn)
