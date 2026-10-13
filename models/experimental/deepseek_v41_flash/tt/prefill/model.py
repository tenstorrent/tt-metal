# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash prefill model: V4's chunked prefill around the V4.1 decoder layers.

Same shape as :class:`~models.experimental.deepseek_v4_flash.tt.model.DeepSeekV4PrefillModel` -- the embedding, the
layer stack (each layer on its pipeline stage's ``1 x TP`` submesh, the residual handed to the next stage through
the host), a head on the last stage -- with what V4.1 changed:

* Layers exchange ``(streams [1, T, hc, D], pre_mix [1, T, 1, hc])``. The first ``pre_mix`` is the one-hot
  ``[1, 0, 0, 0]``. There is no ``hc_head``: the head collapses the last token's streams with the last layer's FFN
  ``pre_mix``, then applies the final RMSNorm and ``lm_head`` (decode's ``_collapse_streams``).
* Engram layers (1 and 14) take the chunk's n-gram rows from an :class:`~..engram_lookup.EngramHostLookup`, called once
  per chunk at the chunk's start position.
* A compressed group's layers share a :class:`~.attention.PrefillShared`. A group can straddle a stage edge, so
  crossing one copies the group's compressed KV and index keys, and the current selection mask, to the next stage
  (through the host, like the streams); those copies live until the chunk is done.

**KV-only prefill** (``kv_only=True``). Layers after the last kv source (``max(kv_source_layer_ids)``, layer 20) only
read that source's compressed KV, so the first 21 layers (0-20) write all of the shared compressed KV and index keys.
This mode builds just those layers, no final norm or ``lm_head``, and :meth:`prefill` returns ``(None, states)``. What it
does not produce: the logits, and the sliding-window rings (and the Engram / hyper-connection state) of layers 21-39,
which depend on the hidden state those layers compute; the caller seeds them another way (decode replays the last
``sliding_window`` tokens, as metal_2's spec path does).

A layer may attend over at most ``candidate_topk_blocks * candidate_block_size`` (16384) compressed entries until
the candidate pre-filter is ported, so a ratio-1 layer caps the prompt at 16384 tokens (see
:class:`~.attention.DeepSeekV41PrefillAttention`).

**Traced prefill** (:class:`DeepSeekV41TracedPrefill`, :meth:`DeepSeekV41PrefillModel.prepare_traced_prefill` /
:meth:`~DeepSeekV41PrefillModel.prefill_traced`) is V4's: one trace per stage and position tier over fixed-size chunks,
the stages pipelined and linked by D2D sockets, no host round trip between them. Its stages start at index sources
(:func:`plan_prefill_placement`).

Handing the states over to decode: :meth:`~..model.DeepSeekV41Model.build_prefill` builds this model on the decode
stages and :meth:`~..model.DeepSeekV41Model.commit_prefill_state` writes its states into the decode caches.
"""

import math
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch
import ttnn
from loguru import logger

from models.experimental.deepseek_v41_flash.tt.config import decoder_layer_weights, dequantized, expert_provider
from models.experimental.deepseek_v41_flash.tt.decode.moe import DeepSeekV41PreloadedExperts
from models.experimental.deepseek_v41_flash.tt.engram_lookup import EngramHostLookup
from models.experimental.deepseek_v41_flash.tt.prefill.attention import (
    PrefillShared,
    PrefillStaticShared,
    ratio_layer_type,
)
from models.experimental.deepseek_v41_flash.tt.prefill.decoder_layer import DeepSeekV41PrefillDecoderLayer
from models.experimental.deepseek_v4_flash.tt.common import _HIFI4, _MASK_NEG
from models.experimental.deepseek_v4_flash.tt.layers import Linear
from models.experimental.deepseek_v4_flash.tt.model import (
    _OUT_SOCKET_CORE,
    _PKT_SOCKET_CORE,
    _PREFILL_PKT_FIFO_PACKETS,
    DeepSeekV4PrefillModel,
    TracedPrefill,
    _create_socket_pair,
    _d2h_page_plan,
    _prefill_tier_factor,
    _round_up,
    _Stage,
)
from models.experimental.deepseek_v4_flash.tt.prefill.attention import (
    ALIGNMENT,
    SLIDING_ATTENTION,
    PrefillAttentionState,
    PrefillStaticStep,
)
from models.experimental.deepseek_v4_flash.tt.prefill.decoder_layer import load_norm_gamma
from models.experimental.deepseek_v4_flash.tt.system_config import active_system_config
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache, _as_cache, _load_weight, _materialize
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader


class DeepSeekV41PrefillModel(DeepSeekV4PrefillModel):
    """ttnn prefill of the V4.1-Flash checkpoint (see the module docstring).

    ``loader`` is the checkpoint; ``device`` holds the embedding and ``layer_devices`` (one per layer, default
    ``device``) place the layers, the head living with the last one. ``tp_size`` is the width of the ``1 x TP``
    meshes. Weight dtypes: ``weight_dtype`` for attention and Engram projections, ``moe_weight_dtype`` for routers and
    shared experts, ``expert_dtype`` for the routed experts this class uploads, ``head_dtype`` for ``lm_head``.
    ``experts``, ``embedding_weight`` and ``lm_head`` reuse tensors another model (decode) already holds.

    ``engram_lookup`` is required when the built layers include an Engram layer.

    ``kv_only`` builds only layers ``0 .. max(kv_source_layer_ids)`` (``num_layers`` defaults to that and may not be
    smaller), skips the head, and makes :meth:`prefill` return ``(None, states)`` (see the module docstring).

    Does not call :meth:`DeepSeekV4PrefillModel.__init__`: it builds V4 layers and an ``hc_head``.
    """

    def __init__(
        self,
        config,
        loader: DeepseekV4WeightLoader,
        device: ttnn.MeshDevice,
        rope: dict,
        num_layers: Optional[int] = None,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        moe_weight_dtype: ttnn.DataType = ttnn.bfloat16,
        expert_dtype: ttnn.DataType = ttnn.bfloat4_b,
        head_dtype: ttnn.DataType = ttnn.bfloat16,
        tp_size: int = 1,
        layer_devices: Optional[Sequence[ttnn.MeshDevice]] = None,
        maskless: bool = False,
        progress: Optional[Callable[..., None]] = None,
        engram_lookup: Optional[EngramHostLookup] = None,
        experts: Optional[Sequence[DeepSeekV41PreloadedExperts]] = None,
        embedding_weight: Optional[ttnn.Tensor] = None,
        lm_head: Optional[Linear] = None,
        kv_only: bool = False,
    ):
        self._progress = progress or (lambda message, important=False: None)
        self._tag = "build"
        note = self._note
        last_source = max(config.kv_source_layer_ids)
        if num_layers is None:
            num_layers = last_source + 1 if kv_only else config.num_hidden_layers
        if kv_only and num_layers <= last_source:
            raise ValueError(f"kv_only needs layers 0..{last_source} (the last kv source), got num_layers {num_layers}")
        if not 0 < num_layers <= config.num_hidden_layers:
            raise ValueError(f"num_layers {num_layers} is not in [1, {config.num_hidden_layers}]")
        if experts is not None and len(experts) < num_layers:
            raise ValueError(f"got {len(experts)} experts objects for {num_layers} layers")
        if layer_devices is None:
            layer_devices = [device] * num_layers
        if len(layer_devices) != num_layers:
            raise ValueError(f"got {len(layer_devices)} layer devices for {num_layers} layers")
        if engram_lookup is None and any(i in config.engram_layer_ids for i in range(num_layers)):
            raise ValueError("the built layers include an Engram layer: pass engram_lookup")
        self.config = config
        self.device = device
        self.layer_devices = list(layer_devices)
        self.head_device = self.layer_devices[-1]
        self.tp_size = tp_size
        self.hidden = config.hidden_size
        self.vocab_size = config.vocab_size
        self.hc = config.hc_mult
        self.engram_lookup = engram_lookup
        self.kv_only = kv_only
        self._traced = None
        self.keep_decode_prefetcher = False
        self.shared_socket_pairs = {}
        # Engram layer -> decode's rows sockets (same core, same one-row page) to reuse instead of opening more.
        self.shared_engram_sockets = {}
        cache = _as_cache(cache)

        if embedding_weight is not None:
            self.embedding_weight = embedding_weight
        else:
            note(f"embedding table [{config.vocab_size} x {config.hidden_size}] -> device", important=True)
            embed_file = cache.file("embed_tokens.prefill")
            embed = _materialize(
                lambda: loader.get_tensor("embed_tokens.weight"), embed_file, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT
            )
            self.embedding_weight = _load_weight(
                embed.detach() if embed is not None else None,
                device,
                cache_file_name=embed_file,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

        self.layers: list[DeepSeekV41PrefillDecoderLayer] = []
        for i in range(num_layers):
            layer_device = self.layer_devices[i]
            layer_cache = cache.sub(f"layers.{i}")
            layer_experts = experts[i] if experts is not None else None
            if layer_experts is None:
                note(f"layer {i + 1}/{num_layers}: routed experts", important=True)
                layer_experts = DeepSeekV41PreloadedExperts(
                    config,
                    self._noting_provider(expert_provider(loader, i), i, num_layers, config.num_local_experts),
                    layer_device,
                    dtype=expert_dtype,
                    cache=layer_cache.sub("ffn"),
                    tp_size=tp_size,
                )
            self.layers.append(
                DeepSeekV41PrefillDecoderLayer(
                    config,
                    i,
                    decoder_layer_weights(loader, i, config),
                    layer_device,
                    rope,
                    layer_experts,
                    cache=layer_cache,
                    weight_dtype=weight_dtype,
                    moe_weight_dtype=moe_weight_dtype,
                    tp_size=tp_size,
                    maskless=maskless,
                )
            )
            note(f"layer {i + 1}/{num_layers}: built")

        self.eps = config.rms_norm_eps
        self.norm_weight = self.lm_head = None
        if not kv_only:
            note("final norm and lm_head -> device", important=True)
            self.norm_weight = load_norm_gamma(
                dequantized(loader, "norm.weight"), self.head_device, cache.file("norm.prefill")
            )
            self.lm_head = (
                lm_head
                if lm_head is not None
                else Linear(
                    dequantized(loader, "lm_head.weight", translate=True),
                    self.head_device,
                    cache.file("lm_head.prefill"),
                    dtype=head_dtype,
                )
            )

    # ------------------------------------------------------------------ pieces
    def _carry(
        self, tensor: ttnn.Tensor, src: ttnn.MeshDevice, dst: ttnn.MeshDevice, layout: ttnn.Layout
    ) -> ttnn.Tensor:
        """A bf16 tensor replicated on ``src`` as a ``layout`` copy replicated on ``dst``, through the host."""
        return ttnn.from_torch(
            self.to_host(tensor, src).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=layout,
            device=dst,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(dst) if dst.get_num_devices() > 1 else None,
        )

    def _initial_pre_mix(self, num_tokens: int) -> ttnn.Tensor:
        """The first layer's ``pre_mix``: the one-hot ``[1, 0, ..., 0]`` per token, ``[1, T, 1, hc]`` bf16 TILE."""
        one_hot = torch.zeros(1, num_tokens, 1, self.hc)
        one_hot[..., 0] = 1
        return ttnn.from_torch(
            one_hot,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device) if self.device.get_num_devices() > 1 else None,
        )

    def _engram_rows(self, ids: torch.Tensor, start: int, device_of: dict) -> dict[int, ttnn.Tensor]:
        """Layer index -> its chunk rows ``[1, 1, T, kin]`` bf16 TILE on that layer's device, from the host lookup."""
        if not any(layer.engram is not None for layer in self.layers):
            return {}
        rows = self.engram_lookup(ids, start)
        return {
            i: ttnn.from_torch(
                rows[i].reshape(1, 1, ids.shape[1], -1),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device_of[i],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device_of[i]) if device_of[i].get_num_devices() > 1 else None,
            )
            for i, layer in enumerate(self.layers)
            if layer.engram is not None
        }

    def _collapse(self, streams: ttnn.Tensor, pre_mix: ttnn.Tensor) -> ttnn.Tensor:
        """``sum_j pre_mix[j] * streams[j]`` in fp32, one stream at a time: ``[1, T, hc, D]`` and ``[1, T, 1, hc]`` ->
        ``[1, T, 1, D]`` bf16. (A ``ttnn.sum`` over ``hc`` in a tile would add the tile's padding.)"""
        t = streams.shape[1]
        streams_f = ttnn.typecast(streams, ttnn.float32)
        weights = ttnn.typecast(pre_mix, ttnn.float32)
        total = None
        for j in range(self.hc):
            row = ttnn.slice(streams_f, [0, 0, j, 0], [1, t, j + 1, self.hidden])
            weight = ttnn.slice(weights, [0, 0, 0, j], [1, t, 1, j + 1])
            term = ttnn.multiply(row, weight)
            total = term if total is None else ttnn.add(total, term)
        return ttnn.typecast(total, ttnn.bfloat16)

    def head(self, streams: ttnn.Tensor, pre_mix: ttnn.Tensor, select: Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        """Final ``(streams [1, T, hc, D], pre_mix [1, T, 1, hc])`` -> one token's logits ``[1, 1, 1, V]``.

        The last token's, or with ``select`` (a one-hot bf16 TILE row ``[1, 1, 1, T]``, traced prefill) the one it
        picks.
        """
        if self.kv_only:
            raise RuntimeError("a kv_only prefill model has no head")
        self._note(f"{self._tag}: head (collapse, final norm, lm_head)")
        t = streams.shape[1]
        if select is None:
            last_streams = ttnn.slice(streams, [0, t - 1, 0, 0], [1, t, self.hc, self.hidden])
            last_mix = ttnn.slice(pre_mix, [0, t - 1, 0, 0], [1, t, 1, self.hc])
        else:
            last_streams = ttnn.reshape(self._select_row(select, streams), [1, 1, self.hc, self.hidden])
            last_mix = self._select_row(select, pre_mix)
        collapsed = self._collapse(last_streams, last_mix)
        ttnn.deallocate(last_streams)
        ttnn.deallocate(last_mix)
        hidden = ttnn.rms_norm(collapsed, weight=self.norm_weight, epsilon=self.eps)
        ttnn.deallocate(collapsed)
        return self.lm_head(hidden)

    @staticmethod
    def _select_row(select: ttnn.Tensor, x: ttnn.Tensor) -> ttnn.Tensor:
        """``x`` ``[1, T, a, b]`` TILE -> the token row ``select`` ``[1, 1, 1, T]`` picks, ``[1, 1, 1, a * b]`` TILE
        (``x`` flattened per token, so the matmul sees no tile padding along ``a``)."""
        t, width = x.shape[1], x.shape[2] * x.shape[3]
        rows = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        flat = ttnn.reshape(rows, [1, 1, t, width])
        tiled = ttnn.to_layout(flat, ttnn.TILE_LAYOUT)
        _free_distinct(rows, flat)
        picked = ttnn.matmul(select, tiled, compute_kernel_config=_HIFI4)
        ttnn.deallocate(tiled)
        return picked

    def _stack(
        self,
        ids: torch.Tensor,
        states: list[PrefillAttentionState],
        on_layer: Optional[Callable[[int, ttnn.Tensor, ttnn.Tensor, ttnn.MeshDevice], None]] = None,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """The embedding and every layer over one chunk of ``ids`` ``[1, T]``: the final ``(streams, pre_mix)``
        on the head device.

        ``on_layer(index, streams, pre_mix, device)`` (verification) is called with each layer's output before it
        moves on; it must not deallocate them.
        """
        if len(states) != len(self.layers):
            raise ValueError(f"expected {len(self.layers)} layer states, got {len(states)}")
        num_tokens = ids.shape[1]
        start = states[0].seq_len
        engram_rows = self._engram_rows(ids, start, dict(enumerate(self.layer_devices)))

        self._note(f"{self._tag}: embedding")
        ids_dev = self._upload_ids(ids, self.device)
        streams = self.embed(ids_dev)
        ttnn.deallocate(ids_dev)
        pre_mix = self._initial_pre_mix(num_tokens)
        shared = PrefillShared()
        mirror = None  # the kv-source copy on the current stage, when the real one lives on an earlier stage
        current = self.device
        stage = 0
        for i, (layer, dev, state) in enumerate(zip(self.layers, self.layer_devices, states)):
            if dev is not current:
                stage += 1
                self._note(
                    f"{self._tag}: stage hand-off to stage {stage} (streams, pre_mix, shared KV -> host -> next)"
                )
                moved_streams = self._carry(streams, current, dev, ttnn.TILE_LAYOUT)
                moved_pre_mix = self._carry(pre_mix, current, dev, ttnn.TILE_LAYOUT)
                ttnn.deallocate(streams)
                ttnn.deallocate(pre_mix)
                streams, pre_mix = moved_streams, moved_pre_mix
                shared, mirror = self._move_shared(shared, mirror, current, dev)
                current = dev
                self._note(f"{self._tag}: stage hand-off done")
            kind = self.config.layer_types[i].replace("_attention", "")
            self._note(f"{self._tag}: layer {i + 1}/{len(self.layers)} ({kind}, stage {stage}) enqueue")
            out, out_pre_mix = layer(streams, pre_mix, state, shared, engram_rows=engram_rows.get(i))
            ttnn.deallocate(streams)
            ttnn.deallocate(pre_mix)
            streams, pre_mix = out, out_pre_mix
            if on_layer is not None:
                on_layer(i, streams, pre_mix, dev)
        for rows in engram_rows.values():
            ttnn.deallocate(rows)
        if shared.entry_sel is not None:
            ttnn.deallocate(shared.entry_sel)
        self._free_mirror(mirror)
        return streams, pre_mix

    def _move_shared(
        self,
        shared: PrefillShared,
        mirror: Optional[PrefillAttentionState],
        src: ttnn.MeshDevice,
        dst: ttnn.MeshDevice,
    ) -> tuple[PrefillShared, Optional[PrefillAttentionState]]:
        """``shared`` for the next stage: the kv source's compressed KV and index keys, and the selection mask, copied
        to ``dst``. ``mirror`` is the copy of a kv source on ``src`` (``None`` when the source itself ran there); it and
        the old mask are freed, the layers' own states are not."""
        moved = PrefillShared()
        new_mirror = None
        source = shared.kv_source
        if source is not None:
            new_mirror = PrefillAttentionState(
                seq_len=source.seq_len,
                compressed_kv=self._carry(source.compressed_kv, src, dst, ttnn.ROW_MAJOR_LAYOUT),
                idx_keys=(
                    None if source.idx_keys is None else self._carry(source.idx_keys, src, dst, ttnn.ROW_MAJOR_LAYOUT)
                ),
            )
            moved.kv_source = new_mirror
        if shared.entry_sel is not None:
            moved.entry_sel = self._carry(shared.entry_sel, src, dst, ttnn.TILE_LAYOUT)
            ttnn.deallocate(shared.entry_sel)
        self._free_mirror(mirror)
        return moved, new_mirror

    @staticmethod
    def _free_mirror(mirror: Optional[PrefillAttentionState]) -> None:
        if mirror is None:
            return
        for tensor in (mirror.compressed_kv, mirror.idx_keys):
            if tensor is not None:
                ttnn.deallocate(tensor)

    # ------------------------------------------------------------------ public
    def forward(self, input_ids, states: Optional[list[PrefillAttentionState]] = None) -> Optional[ttnn.Tensor]:
        """The network over one chunk of a prompt (``[T]`` / ``[1, T]`` ids, ``T`` a multiple of ``ALIGNMENT``).

        ``states`` is this model's per-layer state (:meth:`new_state`), advanced in place; ``None`` runs a whole
        prompt from scratch and discards it. Returns bf16 TILE logits ``[1, 1, 1, V]`` for the chunk's last token,
        on the head device (replicated across its ranks; read one with :meth:`to_host`). ``None`` when ``kv_only``.
        """
        ids = self._host_ids(input_ids)
        self._tag = "forward"
        streams, pre_mix = self._stack(ids, self.new_state() if states is None else states)
        logits = None if self.kv_only else self.head(streams, pre_mix)
        ttnn.deallocate(streams)
        ttnn.deallocate(pre_mix)
        return logits

    def prefill(
        self,
        input_ids,
        chunk_size: int = 1024,
        on_chunk: Optional[Callable[[int, int, int, float], None]] = None,
        states: Optional[list[PrefillAttentionState]] = None,
    ) -> tuple[Optional[ttnn.Tensor], list[PrefillAttentionState]]:
        """A whole prompt as consecutive ``chunk_size`` chunks; returns ``(last-token logits, states)``.

        The logits are ``[1, 1, 1, V]`` for the prompt's last token (``None`` when ``kv_only``). Only the last chunk
        pays for the head.
        ``on_chunk(index, start, end, seconds)`` is called after every chunk with its wall time; giving it makes the
        model synchronize every device before and after each chunk. ``states`` continues an earlier prefill, whose
        prompt must have been the one the Engram lookup last saw up to ``states[0].seq_len``.
        """
        if chunk_size <= 0 or chunk_size % ALIGNMENT:
            raise ValueError(f"chunk_size {chunk_size} must be a positive multiple of {ALIGNMENT}")
        ids = self._host_ids(input_ids)
        total = ids.shape[1]
        states = self.new_state() if states is None else states
        num_chunks = -(-total // chunk_size)
        for index, start in enumerate(range(0, total, chunk_size)):
            end = min(total, start + chunk_size)
            self._tag = f"chunk {index + 1}/{num_chunks}"
            if on_chunk is not None:
                self.synchronize("before the chunk")
            self._note(f"{self._tag} [{start}, {end}): start", important=True)
            t0 = time.perf_counter()
            streams, pre_mix = self._stack(ids[:, start:end], states)
            logits = self.head(streams, pre_mix) if end == total and not self.kv_only else None
            ttnn.deallocate(streams)
            ttnn.deallocate(pre_mix)
            self._note(f"{self._tag}: all work enqueued in {time.perf_counter() - t0:.2f} s")
            if on_chunk is not None:
                self.synchronize("device compute")
                on_chunk(index, start, end, time.perf_counter() - t0)
        return logits, states

    # ------------------------------------------------------------------ traced prefill
    def prepare_traced_prefill(self, max_len: int, chunk_size: int = 1024) -> None:
        """Capture the chunk traces for prompts of up to ``max_len`` tokens (see :class:`DeepSeekV41TracedPrefill`).

        As V4's :meth:`~DeepSeekV4PrefillModel.prepare_traced_prefill`: right after building the model, before any
        eager :meth:`prefill`; one plan at a time.
        """
        if self._traced is not None and self._traced.prepared:
            if self._traced._chunk_size == chunk_size and max_len <= self._traced._max_len:
                return
            raise RuntimeError(
                f"a traced prefill plan for up to {self._traced._max_len} tokens in chunks of "
                f"{self._traced._chunk_size} is prepared; release_traced_prefill() before preparing another"
            )
        self._traced = DeepSeekV41TracedPrefill(self)
        self._traced.prepare(max_len, chunk_size)

    def compile_traced_prefill(self, max_len: int, chunk_size: int = 1024) -> None:
        """The first half of :meth:`prepare_traced_prefill` (no capture); :meth:`capture_traced_prefill` finishes it."""
        if self._traced is not None and (self._traced.prepared or self._traced.compiled):
            raise RuntimeError("a traced prefill plan is already compiled; release_traced_prefill() first")
        self._traced = DeepSeekV41TracedPrefill(self)
        self._traced.compile(max_len, chunk_size)

    def prefill_traced(
        self,
        input_ids,
        on_chunk: Optional[Callable[[int, int, int, float], None]] = None,
    ) -> tuple[Optional[torch.Tensor], list[PrefillAttentionState]]:
        """:meth:`prefill` by replaying the prepared traces, the stages pipelined (see V4's
        :meth:`~DeepSeekV4PrefillModel.prefill_traced`). The logits are host fp32 ``[1, 1, 1, V]``, ``None`` when
        ``kv_only``; the states are views of persistent buffers: consume them, then :meth:`free_traced_states`."""
        if self._traced is None or not self._traced.prepared:
            raise RuntimeError("call prepare_traced_prefill(max_len, chunk_size) first")
        return self._traced.run(input_ids, on_chunk=on_chunk)


def _free_distinct(*tensors: ttnn.Tensor) -> None:
    """Deallocate ``tensors``, once per device buffer: a reshape may return a view of its input or a copy."""
    seen = set()
    for tensor in tensors:
        address = tensor.buffer_address()
        if address not in seen:
            seen.add(address)
            ttnn.deallocate(tensor)


def plan_prefill_placement(config, num_layers: int, num_stages: int) -> list[int]:
    """Stage index of each of the first ``num_layers`` layers, for traced prefill (:class:`DeepSeekV41TracedPrefill`).

    A stage starts only at layer 0 or at an index source, so an index source's selection never crosses a stage edge
    (a stage may still read the compressed KV of a kv source on an earlier stage: it is sent along). The resulting
    segments (``[0, 1]``, ``[2, 7]``, ..., ``[36, 39]``) are merged into at most ``num_stages`` contiguous stages with
    the fewest layers on the busiest one. Returns fewer stages than ``num_stages`` when there are fewer segments.
    """
    starts = sorted({0, *(i for i in config.index_source_layer_ids if i < num_layers)})
    sizes = [hi - lo for lo, hi in zip(starts, starts[1:] + [num_layers])]
    n, k = len(sizes), min(num_stages, len(sizes))
    prefix = [0]
    for size in sizes:
        prefix.append(prefix[-1] + size)
    # best[j][i]: the smallest busiest stage splitting the first i segments into j stages; cut[j][i] its last cut.
    inf = float("inf")
    best = [[inf] * (n + 1) for _ in range(k + 1)]
    cut = [[0] * (n + 1) for _ in range(k + 1)]
    best[0][0] = 0
    for j in range(1, k + 1):
        for i in range(j, n + 1):
            for c in range(j - 1, i):
                cost = max(best[j - 1][c], prefix[i] - prefix[c])
                if cost < best[j][i]:
                    best[j][i], cut[j][i] = cost, c
    bounds, i = [], n
    for j in range(k, 0, -1):
        bounds.append((cut[j][i], i))
        i = cut[j][i]
    placement = []
    for stage, (lo, hi) in enumerate(reversed(bounds)):
        placement.extend([stage] * (prefix[hi] - prefix[lo]))
    return placement


# --- Traced prefill ------------------------------------------------------------------------------ #
# V4's :class:`~models.experimental.deepseek_v4_flash.tt.model.TracedPrefill` (read its section comment first): one
# trace per pipeline stage and position tier, replayed for every chunk with the stages pipelined. A chunk's only host
# input on the critical path is its INT32 packet (H2D on stage 0); the stages hand everything on over D2D sockets and
# the last stage streams its output to the host (D2H). Nothing round-trips through the host between stages.
#
# What V4.1 adds to the D2D hand-off and the persistent state:
#
# * The residual is ``(streams [1, C, hc, D], pre_mix [1, C, 1, hc])``: both go row-major, ``pre_mix`` flattened to
#   rows of ``_PRE_MIX_ROW`` values (a ``[.., 1, hc]`` row would be an 8-byte page). Stage 0's first ``pre_mix`` is a
#   persistent one-hot.
# * Only the kv sources (2, 8, 14, 20) own entry and index-key buffers; the layers of their group read them through a
#   :class:`~.attention.PrefillStaticShared`. Stages start at index sources (:func:`plan_prefill_placement`), so a
#   selection never crosses a stage edge, but a group can: then the source stage also sends the chunk's new entry and
#   key rows (``[1, 1, C / rate, width]``) down the pipeline, and a stage reading that group writes them into its own
#   front-anchored copy (``mirrors``), at the packet's rows like the source does.
# * Engram layers (1, 14) take the chunk's n-gram rows from the host lookup. They are off the critical path: one H2D
#   socket per device of the layer's stage (the device half runs replicated), fed by a writer thread per socket, each
#   replay's ``recv_async_h2d`` taking one chunk's ``C`` row pages. The lookup runs on the host as each packet is fed.
# * ``kv_only``: no head; the last stage sends a one-page "done" token over the D2H socket instead of logits, which
#   keeps V4's flow control (one read per chunk).

_ENGRAM_SOCKET_CORE = (0, 4)  # the k-th Engram layer of a stage uses (0, 4 + k)
# One row page per socket: the FIFO sits in L1 on a worker core whose free top of L1 the prefill ops' circular
# buffers nearly reach (see V4's _PREFILL_PKT_FIFO_PACKETS).
_ENGRAM_FIFO_PAGES = 1
_PRE_MIX_ROW = 64


@dataclass
class _V41StageIO:
    """Persistent per-stage device tensors (see the section comment)."""

    streams_in: Optional[ttnn.Tensor]  # [1, C, hc, D] bf16 ROW_MAJOR (stages after the first)
    pre_in: Optional[ttnn.Tensor]  # [1, 1, C * hc / _PRE_MIX_ROW, _PRE_MIX_ROW] bf16 ROW_MAJOR (stages after the first)
    one_hot: Optional[ttnn.Tensor]  # [1, C, 1, hc] bf16 TILE, the first pre_mix (stage 0)
    masks: dict  # layer type -> (static, threshold), as V4's _StageIO
    index_pads: dict  # (layer type, tier cap) -> [1, 1, kv_len, index_head_dim] zero key pad
    ramp: Optional[ttnn.Tensor]  # [1, 1, 1, C] fp32 TILE 0 .. C - 1 (last stage with a head)
    rows_in: dict  # remote kv source -> (entries, keys) [1, 1, C / rate, width] bf16 ROW_MAJOR, received
    mirrors: dict  # remote kv source -> (entries, keys) [1, 1, capacity, width] bf16 ROW_MAJOR, front-anchored
    engram_rows: dict  # Engram layer -> [1, 1, C, kin] bf16 ROW_MAJOR, received from the host


@dataclass
class _V41Stage(_Stage):
    sources: set = field(default_factory=set)  # the kv sources on this stage
    indexed: set = field(default_factory=set)  # layer types of this stage's index sources
    mirrored: list = field(default_factory=list)  # remote kv sources this stage's layers read
    carried_in: list = field(default_factory=list)  # remote kv sources whose new rows arrive from the previous stage
    carried_out: list = field(default_factory=list)  # kv sources whose new rows go on to the next stage
    engram_layers: list = field(default_factory=list)
    engram_sockets: dict = field(default_factory=dict)  # Engram layer -> one H2DSocket per device


class DeepSeekV41TracedPrefill(TracedPrefill):
    """Traced, pipelined execution of a :class:`DeepSeekV41PrefillModel` (see the section comment above)."""

    def __init__(self, model: DeepSeekV41PrefillModel):
        super().__init__(model)
        config = self.config
        n = model.num_layers
        self._kv_sources = sorted(i for i in config.kv_source_layer_ids if i < n)
        self._index_sources = sorted(i for i in config.index_source_layer_ids if i < n)
        self._key_rate = {
            ratio_layer_type(r): r for r in {config.compress_ratios[i] for i in self._kv_sources}
        }  # layer type -> compress rate
        self._done_w = 0  # INT32 slots of the kv_only "done" page
        self._engram_queues: dict = {}  # Engram layer -> one host-rows queue per socket
        self._engram_threads: list = []

    # ------------------------------------------------------------------ planning
    def _source_of(self, layer_idx: int) -> int:
        """The kv source whose entries compressed layer ``layer_idx`` reads."""
        return max(i for i in self._kv_sources if i <= layer_idx)

    def _index_source_of(self, layer_idx: int) -> int:
        """The index source whose selection compressed layer ``layer_idx`` uses."""
        return max(i for i in self._index_sources if i <= layer_idx)

    def _caps_at(self, chunks: int, chunk_size: int) -> dict:
        return {
            key: min(full, _round_up(max(chunks * chunk_size // self._key_rate[key], 1), ALIGNMENT))
            for key, full in self._entry_capacity.items()
        }

    def _rates(self, stage: _V41Stage) -> list[int]:
        return sorted({self._key_rate[key] for key in stage.types if key != SLIDING_ATTENTION})

    def _representative(self, stage: _V41Stage, layer_type: str):
        for li in stage.layers:
            if self.model.layers[li].self_attn.layer_type == layer_type:
                return self.model.layers[li].self_attn
        raise KeyError(layer_type)

    def _group_stages(self) -> list[_V41Stage]:
        model = self.model
        stages: list[_V41Stage] = []
        for li, dev in enumerate(model.layer_devices):
            if not stages or stages[-1].device is not dev:
                stages.append(_V41Stage(index=len(stages), device=dev, layers=[], first=not stages))
            stages[-1].layers.append(li)
            stages[-1].types.add(model.layers[li].self_attn.layer_type)
        stages[-1].last = True
        stage_of = {li: stage.index for stage in stages for li in stage.layers}
        needed = {}
        for stage in stages:
            stage.sources = {li for li in stage.layers if li in self._kv_sources}
            compressed = [li for li in stage.layers if model.layers[li].self_attn.rate]
            for li in compressed:
                if stage_of[self._index_source_of(li)] != stage.index:
                    raise NotImplementedError(
                        f"layer {li} is on stage {stage.index} but its index source {self._index_source_of(li)} is "
                        "not: start stages at index sources (plan_prefill_placement)"
                    )
            stage.indexed = {model.layers[li].self_attn.layer_type for li in compressed if li in self._index_sources}
            needed[stage.index] = sorted({self._source_of(li) for li in compressed} - stage.sources)
            stage.mirrored = needed[stage.index]
            stage.engram_layers = [li for li in stage.layers if model.layers[li].engram is not None]
        for stage in stages:
            later = {src for s in stages[stage.index :] for src in needed[s.index]}
            stage.carried_in = sorted(src for src in later if stage_of[src] < stage.index)
            after = {src for s in stages[stage.index + 1 :] for src in needed[s.index]}
            stage.carried_out = sorted(src for src in after if stage_of[src] <= stage.index)
        return stages

    def _layout_packet(self, chunk_size: int) -> None:
        """V4's packet (:meth:`TracedPrefill._layout_packet`) for the kv sources' rates. With ``kv_only`` the "done"
        page is the first ``_done_w`` position slots (the chunk's start first)."""
        offset = 2 * chunk_size
        self._pkt_entry, self._pkt_rows = {}, {}
        for rate in sorted(set(self._key_rate.values())):
            self._pkt_entry[rate] = offset
            offset += chunk_size // rate
            self._pkt_rows[rate] = offset
            offset += chunk_size // rate
        self._pkt_last = offset
        offset += 1
        alignment = active_system_config().pipeline.pcie_alignment
        self._pkt_page_bytes = math.ceil(offset * 4 / alignment) * alignment
        self._pkt_w = self._pkt_page_bytes // 4
        self._pkt_bcast_width = max(d for d in range(1, min(self._pkt_w, 1024) + 1) if self._pkt_w % d == 0)
        self._done_w = alignment // 4

    # ------------------------------------------------------------------ allocation (before any trace)
    def _allocate(self, max_len: int, chunk_size: int) -> None:
        model = self.model
        padded = _round_up(max_len, chunk_size)  # the last chunk's padding emits entries too
        self._entry_capacity = {
            key: _round_up(max(padded // rate, 1), ALIGNMENT) for key, rate in self._key_rate.items()
        }
        self.stages = self._group_stages()
        self._layout_packet(chunk_size)
        self._tiers = self._plan_tiers(padded // chunk_size, _prefill_tier_factor())
        self._tier_caps = [self._caps_at(chunks, chunk_size) for chunks in self._tiers]
        for stage in self.stages:
            stage.tier_sigs = [self._tier_sig(stage, tier) for tier in range(len(self._tiers))]

        for li, layer in enumerate(model.layers):
            attn = layer.self_attn
            self.buffers[li] = attn.new_static_buffers(chunk_size, self._entry_capacity.get(attn.layer_type, 0))

        for stage in self.stages:
            dev = stage.device
            stage.pkt = ttnn.from_torch(
                torch.zeros(1, 1, 1, self._pkt_w, dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(dev) if dev.get_num_devices() > 1 else None,
            )
            for key in stage.types:
                attn = self._representative(stage, key)
                if attn.rope_kind not in stage.rope:
                    stage.rope[attn.rope_kind] = self._rope_tables(attn, max_len, padded)
            stage.io = self._allocate_io(stage, chunk_size)
        self._open_sockets()

    def _allocate_io(self, stage: _V41Stage, t: int) -> _V41StageIO:
        model, config = self.model, self.config
        dev = stage.device

        def zeros(shape: tuple, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, value=None) -> ttnn.Tensor:
            return ttnn.from_torch(
                torch.zeros(shape) if value is None else value,
                dtype=dtype,
                layout=layout,
                device=dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(dev) if dev.get_num_devices() > 1 else None,
            )

        hc = model.hc
        if t * hc % _PRE_MIX_ROW:
            raise ValueError(f"chunk_size {t} x hc {hc} is not a multiple of {_PRE_MIX_ROW}")
        streams_in = pre_in = one_hot = None
        if stage.first:
            hot = torch.zeros(1, t, 1, hc)
            hot[..., 0] = 1
            one_hot = zeros(hot.shape, ttnn.TILE_LAYOUT, value=hot)
        else:
            streams_in = zeros((1, t, hc, model.hidden))
            pre_in = zeros((1, 1, t * hc // _PRE_MIX_ROW, _PRE_MIX_ROW))

        masks, index_pads = {}, {}
        for key in stage.types:
            attn = self._representative(stage, key)
            if attn.maskless:
                continue
            static, threshold = attn.mask_tables_host(t, self._entry_capacity.get(key, 0))
            masks[key] = (
                attn._to_device(static.reshape(1, 1, t, -1)),
                attn._to_device(threshold.reshape(1, 1, t, -1), dtype=ttnn.float32),
            )
            if key in stage.indexed:
                # One pad per distinct tier cap: a pad's rows past its cap stay zero for ever.
                for tier_cap in sorted({caps.get(key, 0) for caps in self._tier_caps}):
                    index_pads[(key, tier_cap)] = attn._zeros_rm(_round_up(tier_cap + t, 64), config.index_head_dim)
        ramp = None
        if stage.last and not model.kv_only:
            ramp = zeros(
                (1, 1, 1, t), ttnn.TILE_LAYOUT, ttnn.float32, torch.arange(t, dtype=torch.float32).reshape(1, 1, 1, t)
            )

        def group_rows(src: int, rows: int) -> tuple:
            return zeros((1, 1, rows, model.layers[src].self_attn.head_dim)), zeros((1, 1, rows, config.index_head_dim))

        rate_of = {src: config.compress_ratios[src] for src in self._kv_sources}
        rows_in = {src: group_rows(src, t // rate_of[src]) for src in stage.carried_in}
        mirrors = {src: group_rows(src, self._entry_capacity[ratio_layer_type(rate_of[src])]) for src in stage.mirrored}
        engram_rows = {li: zeros((1, 1, t, model.layers[li].engram.kin)) for li in stage.engram_layers}
        return _V41StageIO(streams_in, pre_in, one_hot, masks, index_pads, ramp, rows_in, mirrors, engram_rows)

    def _open_sockets(self) -> None:
        """V4's packet, D2D and D2H sockets (the D2H page is the "done" token when ``kv_only``), and one Engram H2D
        socket per device of each Engram layer's stage, each with its writer thread."""
        pipeline = active_system_config().pipeline
        first, last = self.stages[0], self.stages[-1]
        self._pkt_socket = ttnn.H2DSocket(
            first.device,
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(*_PKT_SOCKET_CORE)),
            ttnn.BufferType.L1,
            _PREFILL_PKT_FIFO_PACKETS * self._pkt_page_bytes,
            ttnn.H2DMode.HOST_PUSH,
        )
        self._pkt_socket.set_page_size(self._pkt_page_bytes)
        for upstream, downstream in zip(self.stages, self.stages[1:]):
            pair = self.model.shared_socket_pairs.get((id(upstream.device), id(downstream.device)))
            if pair is None:
                pair = _create_socket_pair(upstream.device, downstream.device, pipeline.socket_l1_bytes)
            upstream.send, downstream.recv = pair
        self._out_socket = ttnn.D2HSocket(
            last.device,
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(*_OUT_SOCKET_CORE)),
            pipeline.d2h_fifo_bytes,
        )
        if self.model.kv_only:
            self._out_socket.set_page_size(self._done_w * 4)
        else:
            self._out_plan = _d2h_page_plan(
                self.model.vocab_size,
                2,
                page_cap_bytes=pipeline.d2h_fifo_bytes,
                pcie_alignment=pipeline.pcie_alignment,
            )
            self._out_socket.set_page_size(self._out_plan[1] * 2)

        for stage in self.stages:
            for k, li in enumerate(stage.engram_layers):
                page_bytes = self.model.layers[li].engram.kin * 2  # one bf16 row
                if page_bytes % pipeline.pcie_alignment:
                    raise ValueError(f"a {page_bytes} B Engram row is not a multiple of the PCIe alignment")
                sockets = self.model.shared_engram_sockets.get(li)
                if sockets is None:
                    core = ttnn.CoreCoord(_ENGRAM_SOCKET_CORE[0], _ENGRAM_SOCKET_CORE[1] + k)
                    sockets = [
                        ttnn.H2DSocket(
                            stage.device,
                            ttnn.MeshCoreCoord(coord, core),
                            ttnn.BufferType.L1,
                            _ENGRAM_FIFO_PAGES * page_bytes,
                            ttnn.H2DMode.HOST_PUSH,
                        )
                        for coord in ttnn.MeshCoordinateRange(stage.device.shape)
                    ]
                    for socket in sockets:
                        socket.set_page_size(page_bytes)
                stage.engram_sockets[li] = sockets
                self._engram_queues[li] = []
                for d, socket in enumerate(sockets):
                    rows_queue: queue.Queue = queue.Queue()
                    thread = threading.Thread(
                        target=self._engram_writer,
                        args=(socket, rows_queue),
                        name=f"prefill-engram-{li}-{d}",
                        daemon=True,
                    )
                    thread.start()
                    self._engram_queues[li].append(rows_queue)
                    self._engram_threads.append((rows_queue, thread))

    @staticmethod
    def _engram_writer(socket, rows_queue: queue.Queue) -> None:
        """Writer-thread body: push each queued chunk's rows into ``socket`` (blocks while its FIFO is full)."""
        try:
            for rows in iter(rows_queue.get, None):
                socket.write_tensor(rows)
        except Exception:
            logger.exception("[traced-prefill] Engram writer failed; the stage will wait on its socket forever")
            raise

    def _feed_engram(self, chunk_ids: Optional[torch.Tensor], start: int) -> None:
        """Queue one chunk's Engram rows ``[C, kin]`` for every Engram socket: the host lookup of ``chunk_ids`` (its
        ``t <= C`` real tokens) at ``start``, zero past them; all zero when ``chunk_ids`` is ``None`` (compile and
        unwind runs). Queue before the chunk's packet: its replay waits on the rows."""
        if not self._engram_queues:
            return
        c = self._chunk_size
        found = None if chunk_ids is None else self.model.engram_lookup(chunk_ids.reshape(1, -1), start)
        for li, queues in self._engram_queues.items():
            rows = torch.zeros(c, self.model.layers[li].engram.kin, dtype=torch.bfloat16)
            if found is not None:
                t = chunk_ids.numel()
                rows[:t] = found[li].reshape(t, -1)
            # The socket copies raw bytes; an integer view keeps bf16 out of the DLPack dtype check.
            rows = rows.view(torch.int16)
            for rows_queue in queues:
                rows_queue.put(rows)

    # ------------------------------------------------------------------ the traced body
    def _step_inputs(self, stage: _V41Stage, pkt: ttnn.Tensor, tier: int = 0) -> tuple:
        """V4's :meth:`TracedPrefill._step_inputs`, keyed by V4.1's layer types (one per compress rate)."""
        t = c = self._chunk_size
        io = stage.io
        caps = {key: self._tier_caps[tier].get(key, 0) for key in stage.types}
        sw = self.config.sliding_window
        made: list = []

        def keep(tensor: ttnn.Tensor) -> ttnn.Tensor:
            made.append(tensor)
            return tensor

        def slots(offset: int, n: int) -> ttnn.Tensor:
            row = ttnn.slice(pkt, [0, 0, 0, offset], [1, 1, 1, offset + n])
            return keep(ttnn.typecast(ttnn.reshape(row, [1, n]), ttnn.uint32))

        def gather(tables: tuple, positions: ttnn.Tensor, n: int) -> tuple:
            out = []
            for table in tables:
                rows = ttnn.embedding(positions, table, layout=ttnn.ROW_MAJOR_LAYOUT)  # [1, n, rope_dim]
                out.append(keep(ttnn.to_layout(ttnn.reshape(rows, [1, 1, n, rows.shape[-1]]), ttnn.TILE_LAYOUT)))
                ttnn.deallocate(rows)
            return tuple(out)

        ids = slots(0, t)
        positions = slots(c, t)
        rope = {kind: gather(tables, positions, t) for kind, tables in stage.rope.items()}
        local_rates = {self.model.layers[li].self_attn.rate for li in stage.sources}
        entry_rope, entry_rows = {}, {}
        for rate in self._rates(stage):
            n = t // rate
            if rate in local_rates:
                entry_rope[rate] = gather(stage.rope["compress"], slots(self._pkt_entry[rate], n), n)
            entry_rows[rate] = ttnn.reshape(slots(self._pkt_rows[rate], n), [n])  # a view: slots() frees it

        masks, index_cuts, index_pads = {}, {}, {}
        start_f = None
        if io.masks:
            start = ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, c], [1, 1, 1, c + 1]), [1, 1, 1, 1])
            start_f = keep(ttnn.typecast(ttnn.to_layout(start, ttnn.TILE_LAYOUT), ttnn.float32))
        for key in stage.types:
            if key not in io.masks:  # mask-free sliding layer
                continue
            cap = caps[key]
            static, threshold = io.masks[key]
            if cap < self._entry_capacity.get(key, 0):
                cols = [1, 1, t, sw + t + cap]
                static = keep(ttnn.slice(static, [0, 0, 0, 0], cols))
                threshold = keep(ttnn.slice(threshold, [0, 0, 0, 0], cols))
            cut = ttnn.typecast(ttnn.multiply(ttnn.gt(threshold, start_f), _MASK_NEG), ttnn.bfloat16)
            masks[key] = keep(ttnn.add(static, cut))
            ttnn.deallocate(cut)
            if key in stage.indexed:
                index_cuts[key] = keep(ttnn.slice(masks[key], [0, 0, 0, sw + t], [1, 1, t, sw + t + cap]))
                index_pads[key] = io.index_pads[(key, cap)]
        step = PrefillStaticStep(
            rope=rope,
            entry_rope=entry_rope,
            masks=masks,
            caps=caps,
            index_cuts=index_cuts,
            index_pads=index_pads,
            entry_rows=entry_rows,
            first_chunk=tier == 0,
        )
        return ids, step, made

    def _stage_forward(self, stage: _V41Stage, tier: int = 0) -> None:
        """One stage over one chunk of position tier ``tier`` (shared by the compile run and the capture).

        Receives (stage 0: the packet from the host; later: streams, pre_mix, the carried groups' new rows, the
        packet, in send order) and the Engram rows, updates the stage's mirrors, runs its layers, and sends the
        same on (the last stage: the logits, or the "done" token).
        """
        model = self.model
        io = stage.io
        pkt = stage.pkt
        c, hc = self._chunk_size, model.hc
        if stage.first:
            ttnn.experimental.recv_async_h2d(stage.pkt, self._pkt_socket)
            if stage.device.get_num_devices() > 1:
                # ``broadcast`` corrupts a row-major page larger than one fabric packet (~4.3 KB): send rows of <= 4 KB.
                width = self._pkt_bcast_width
                rows = ttnn.reshape(stage.pkt, [1, 1, self._pkt_w // width, width])
                sent = ttnn.broadcast(rows, ttnn.MeshCoordinate(0, 0), cluster_axis=1, topology=ttnn.Topology.Ring)
                pkt = ttnn.reshape(sent, [1, 1, 1, self._pkt_w])
        else:
            ttnn.experimental.recv_direct_async(io.streams_in, stage.recv)
            ttnn.experimental.recv_direct_async(io.pre_in, stage.recv)
            for src in stage.carried_in:
                for tensor in io.rows_in[src]:
                    ttnn.experimental.recv_direct_async(tensor, stage.recv)
            ttnn.experimental.recv_direct_async(stage.pkt, stage.recv)
        for li in stage.engram_layers:
            for socket in stage.engram_sockets[li]:
                ttnn.experimental.recv_async_h2d(io.engram_rows[li], socket)
        ids, step, made = self._step_inputs(stage, pkt, tier)

        for src, mirror in io.mirrors.items():
            rate = self.config.compress_ratios[src]
            key = ratio_layer_type(rate)
            writer = self._representative(stage, key)
            for buf, new in zip(mirror, io.rows_in[src]):
                writer._write_entries(buf, ttnn.clone(new), step.entry_rows[rate], step.caps[key])

        if stage.first:
            streams, pre_mix = model.embed(ids), io.one_hot
        else:
            streams = ttnn.to_layout(io.streams_in, ttnn.TILE_LAYOUT)
            view = ttnn.reshape(io.pre_in, [1, c, 1, hc])
            pre_mix = ttnn.to_layout(view, ttnn.TILE_LAYOUT)
            if view.buffer_address() != io.pre_in.buffer_address():
                ttnn.deallocate(view)
        shared = PrefillStaticShared()
        new_rows = {}
        for li in stage.layers:
            layer = model.layers[li]
            attn = layer.self_attn
            if attn.rate and not attn.is_kv_source and self._source_of(li) in io.mirrors:
                shared.entries, shared.idx_keys = io.mirrors[self._source_of(li)]
            shared.keep_rows = li in stage.carried_out
            engram_rows = None
            if li in io.engram_rows:
                engram_rows = ttnn.to_layout(io.engram_rows[li], ttnn.TILE_LAYOUT)
            out, out_pre = layer.forward_static(streams, pre_mix, self.buffers[li], step, shared, engram_rows)
            if engram_rows is not None:
                ttnn.deallocate(engram_rows)
            if shared.new_rows is not None:
                new_rows[li], shared.new_rows = shared.new_rows, None
            ttnn.deallocate(streams)
            if pre_mix is not io.one_hot:
                ttnn.deallocate(pre_mix)
            streams, pre_mix = out, out_pre
        if shared.entry_sel is not None:
            ttnn.deallocate(shared.entry_sel)

        if stage.last:
            if model.kv_only:
                done = ttnn.reshape(ttnn.slice(pkt, [0, 0, 0, c], [1, 1, 1, c + self._done_w]), [1, self._done_w])
                ttnn.experimental.send_async_d2h(done, self._out_socket)
                ttnn.deallocate(done)
            else:
                select = self._head_select(stage, pkt)
                self._send_logits(model.head(streams, pre_mix, select=select))
                ttnn.deallocate(select)
        else:
            rows = ttnn.to_layout(streams, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.experimental.send_direct_async(rows, stage.send)
            pre_rows = ttnn.to_layout(pre_mix, ttnn.ROW_MAJOR_LAYOUT)
            pre_flat = ttnn.reshape(pre_rows, [1, 1, c * hc // _PRE_MIX_ROW, _PRE_MIX_ROW])
            ttnn.experimental.send_direct_async(pre_flat, stage.send)
            for src in stage.carried_out:
                for tensor in new_rows[src] if src in new_rows else io.rows_in[src]:
                    ttnn.experimental.send_direct_async(tensor, stage.send)
            ttnn.experimental.send_direct_async(pkt, stage.send)
            ttnn.deallocate(rows)
            _free_distinct(pre_rows, pre_flat)
        for tensors in new_rows.values():
            for tensor in tensors:
                ttnn.deallocate(tensor)
        ttnn.deallocate(streams)
        if pre_mix is not io.one_hot:
            ttnn.deallocate(pre_mix)
        for tensor in made:
            ttnn.deallocate(tensor)
        if pkt is not stage.pkt:
            ttnn.deallocate(pkt)

    def _read_logits(self) -> Optional[torch.Tensor]:
        """The oldest unread chunk's output off the D2H socket: V4's logits, or ``None`` (the "done" token)."""
        if not self.model.kv_only:
            return super()._read_logits()
        self._out_socket.read_tensor(torch.empty(1, self._done_w, dtype=torch.int32))
        return None

    # ------------------------------------------------------------------ compile / run / release
    def compile(self, max_len: int, chunk_size: int = 1024) -> None:
        """V4's :meth:`TracedPrefill.compile`, also feeding the compile run's (zero) Engram rows."""
        model = self.model
        for name, value in (("max_len", max_len), ("chunk_size", chunk_size)):
            if value <= 0 or value % ALIGNMENT:
                raise ValueError(f"{name}={value} must be a positive multiple of {ALIGNMENT}")
        if self.prepared or self.compiled:
            raise RuntimeError("traced prefill is already prepared; another plan needs another TracedPrefill")
        if any(layer.self_attn.maskless for layer in model.layers) and chunk_size < self.config.sliding_window:
            raise ValueError(
                f"mask-free traced prefill needs chunk_size >= sliding_window ({self.config.sliding_window}), "
                f"got {chunk_size}"
            )
        self._max_len, self._chunk_size = max_len, chunk_size
        model._tag = "traced prepare"
        model._note("traced prefill: allocating persistent buffers and sockets", important=True)
        self._allocate(max_len, chunk_size)
        self._stop_prefetcher()
        tiers = self._distinct_tiers()
        logger.info(
            f"[traced-prefill] prompts of up to {max_len} tokens in chunks of {chunk_size}: {len(tiers)} position "
            f"tier(s) x {len(self.stages)} stage(s) (layers {[s.layers[0] for s in self.stages]}...), chunk counts "
            f"{self._tiers}; entry rows per tier {self._tier_caps}; buffers {self._entry_capacity}; carried groups "
            f"{[s.carried_out for s in self.stages]}; packet {self._pkt_page_bytes} B"
        )
        for tier in tiers:
            self._feed_engram(None, 0)
            self._pkt_socket.write_tensor(self._packet(torch.zeros(chunk_size, dtype=torch.long), 0))
            for stage in self.stages:
                model._note(f"traced prefill: compiling stage {stage.index} (tier {tier})", important=True)
                self._stage_forward(stage, tier)
            self._read_logits()
        for stage in self.stages:
            ttnn.synchronize_device(stage.device)
        self.compiled = True

    def run(
        self,
        input_ids,
        on_chunk: Optional[Callable[[int, int, int, float], None]] = None,
    ) -> tuple[Optional[torch.Tensor], list[PrefillAttentionState]]:
        """V4's :meth:`TracedPrefill.run`, queueing each chunk's Engram rows ahead of its packet. Returns
        ``(logits or None when kv_only, states)``."""
        if not self.prepared:
            raise RuntimeError("call prepare(max_len, chunk_size) first")
        model = self.model
        ids = model._host_ids(input_ids)
        n = ids.shape[1]
        if n <= 0 or n % ALIGNMENT or n > self._max_len:
            raise ValueError(
                f"traces were captured for prompts of up to {self._max_len} tokens in multiples of {ALIGNMENT}, got {n}"
            )
        if model.keep_decode_prefetcher:
            for device in self._devices():
                ttnn.synchronize_device(device)
        else:
            self._stop_prefetcher()
        self._reset()
        self._prompt_len = n
        plan = self._plan_chunks(n, self._chunk_size)
        num_chunks = len(plan)
        in_flight = len(self.stages) + _PREFILL_PKT_FIFO_PACKETS - 1
        ahead = in_flight + len(self.stages)
        model._tag = "traced prefill"
        model._note(
            f"traced prefill: {num_chunks} chunk(s) through {len(self.stages)} stage(s), up to {in_flight} in flight",
            important=True,
        )

        replay: queue.Queue = queue.Queue()
        thread = threading.Thread(target=self._replay, args=(replay,), name="prefill-replay", daemon=True)
        thread.start()
        posted = fed = read = 0
        logits = None
        last_arrival = time.perf_counter()

        def feed(index: int, chunk_ids: Optional[torch.Tensor]) -> None:
            nonlocal fed
            start, t = plan[index]
            self._feed_engram(chunk_ids, start)
            real = chunk_ids if chunk_ids is not None else torch.zeros(t, dtype=torch.long)
            self._pkt_socket.write_tensor(self._packet(real, start))
            fed += 1

        def take() -> None:
            nonlocal read, logits, last_arrival
            logits = self._read_logits()
            read += 1
            now = time.perf_counter()
            start, t = plan[read - 1]
            model._note(f"traced chunk {read}/{num_chunks} [{start}, {start + t}): output received")
            if on_chunk is not None:
                on_chunk(read - 1, start, start + t, now - last_arrival)
            last_arrival = now

        try:
            for index, (start, t) in enumerate(plan):
                while posted < min(num_chunks, index + ahead):
                    replay.put(posted)
                    posted += 1
                feed(index, ids[0, start : start + t])
                while fed - read > in_flight:
                    take()
            while read < fed:
                take()
        finally:
            try:
                # An error part-way leaves posted replays parked on their in-trace receives: feed each a dummy chunk
                # and drain its output, so the sockets and the replay thread can unwind.
                while read < posted:
                    if read >= fed:
                        feed(read, None)
                    self._read_logits()
                    read += 1
            finally:
                replay.put(None)
                thread.join()
        model.synchronize("traced prefill done")
        return logits, self._export_states()

    def release(self) -> None:
        """V4's :meth:`TracedPrefill.release`, also stopping the Engram writers and closing their sockets."""
        for rows_queue, _ in self._engram_threads:
            rows_queue.put(None)
        for _, thread in self._engram_threads:
            thread.join()
        self._engram_threads, self._engram_queues = [], {}
        for stage in self.stages:
            stage.engram_sockets = {}
        super().release()

    def _export_states(self) -> list[PrefillAttentionState]:
        """Per-layer :class:`PrefillAttentionState` of the prompt just run, as the eager :meth:`prefill` leaves it:
        every layer's ``kv_tail`` (cut out of the last chunk's window when it was padded), a kv source's first
        ``prompt_len // rate`` entry and index-key rows. Free with :meth:`free_states` before the next run."""
        n, c = self._prompt_len, self._chunk_size
        pad = _round_up(n, c) - n
        real = c - pad

        def rows(t: ttnn.Tensor, lo: int, hi: int) -> ttnn.Tensor:
            return ttnn.slice(t, [0, 0, lo, 0], [1, 1, hi, t.shape[3]])

        states = []
        for li, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            bufs = self.buffers[li]
            tail = rows(bufs.window, real, real + attn.sliding_window) if pad else bufs.tail
            state = PrefillAttentionState(seq_len=n, kv_tail=tail)
            if attn.is_kv_source:
                entries = n // attn.rate
                state.compressed_kv = rows(bufs.entries, 0, entries)
                state.idx_keys = rows(bufs.idx_keys, 0, entries)
            states.append(state)
        return states
