# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Prefill model for DeepSeek-V4-Flash: the whole network over a prompt, many tokens per call.

:class:`DeepSeekV4PrefillModel` is the prefill counterpart of the decode
:class:`~..model.DeepSeekV4Model`, following the reference ``DeepseekV4ForCausalLM``::

    streams = embed_tokens(ids) expanded to hc parallel streams        # [1, T, hc, D]
    for layer in layers: streams = layer(streams)                      # PrefillDecoderLayer
    hidden  = norm(hc_head(streams))                                   # collapse the streams, RMSNorm
    logits  = lm_head(hidden)

It composes :class:`~.decoder_layer.DeepSeekV4PrefillDecoderLayer` (which composes the prefill attention,
MoE and hyper-connection blocks) with the embedding, a prefill :class:`DeepSeekV4PrefillHyperHead`, the
final RMSNorm and ``lm_head``. Like every prefill block it takes prompts whose length is a multiple of
``ALIGNMENT`` (128); a ragged tail is left to the decode path.

Devices. ``tp_size == 1`` runs on one device. With ``tp_size > 1`` every layer lives on a ``1 x tp_size``
mesh (attention heads and the MoE intermediate width sharded over its ranks, everything else replicated).
``layer_devices`` places each layer on such a mesh, which is how the real 43-layer checkpoint fits: the
routed experts alone are ~20 GB per chip per half of the stack, so the layers are split over two
pipeline stages (two ``1 x 4`` submeshes of an 8-chip mesh), the embedding sits on the first stage and the
head on the last. The residual streams cross a stage boundary through the host (one ``[1, T, hc, D]``
tensor per chunk per boundary), which is simple and off the critical path of the per-layer compute.

A prompt is fed either whole or as consecutive chunks through one list of per-layer
:class:`~.attention.PrefillAttentionState` (:meth:`DeepSeekV4PrefillModel.new_state`); :meth:`prefill`
does the chunking. Logits are computed for the last token only by default -- the one row a prompt's first
generated token needs -- so a long prompt never materialises a ``[T, vocab]`` tensor.

Weights use the checkpoint's names (no ``model.`` prefix): ``embed_tokens.weight``,
``layers.{i}.<decoder-layer keys>``, ``hc_head.{hc_fn,hc_base,hc_scale}``, ``norm.weight`` and
``lm_head.weight``, each a torch tensor or a zero-arg thunk. The routed experts are not in that dict:
``expert_provider(layer_idx)`` returns the per-expert ``provider(e) -> (gate_up [2I, D], down [D, I])``
that :class:`~..decode.moe.DeepSeekV4PreloadedExperts` uploads, or ready-made ``experts`` are handed in
(decode already owns them, and prefill reads the same weights in place).
"""

import time
from typing import Callable, Optional, Sequence

import torch

import ttnn

from ..common import DeepSeekV4Module
from ..decode.decoder_layer import _strip_prefix
from ..decode.moe import DeepSeekV4PreloadedExperts
from ..layers import Linear
from ..weight_cache import WeightCache, _as_cache, _load_weight, _materialize
from .attention import ALIGNMENT, PrefillAttentionState
from .decoder_layer import DeepSeekV4PrefillDecoderLayer, load_norm_gamma
from .hyperconnection import flatten_streams


class DeepSeekV4PrefillHyperHead(DeepSeekV4Module):
    """ttnn prefill port of ``DeepseekV4HyperHead`` (the final stream collapse).

    The multi-token counterpart of the decode :class:`~..decode.hyperconnection.DeepSeekV4HyperHead`::

        flat = unweighted_rmsnorm(streams.flatten(2))
        pre  = sigmoid(hc_fn @ flat * hc_scale + hc_base) + eps
        out  = (pre[..., None] * streams).sum(dim=2)

    Unlike the layers' hyper-connections there is no ``post`` / ``comb``: the head only produces the
    collapsed sequence. ``weights`` keys: ``hc_fn`` ``[hc, hc*D]``, ``hc_base`` ``[hc]``, ``hc_scale``
    (a single scalar). The decode head keeps its operands in width-sharded L1, which is sized for a few
    token rows; this one runs on DRAM-interleaved tensors.
    """

    def __init__(
        self,
        config,
        weights: dict,
        device: ttnn.MeshDevice,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        self.device = device
        self.hc = config.hc_mult
        self.hidden = config.hidden_size
        self.eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        cache = _as_cache(cache)

        self.fn = Linear(weights["hc_fn"], device, cache.file("hc_fn.prefill"), dtype=weight_dtype)  # [hc, hc*D]
        base_src = weights["hc_base"]
        base_file = cache.file("hc_base.prefill")
        base = _materialize(
            lambda: (base_src() if callable(base_src) else base_src).detach().reshape(1, 1, 1, self.hc),
            base_file,
            ttnn.bfloat16,
        )
        self.base = _load_weight(base, device, cache_file_name=base_file)
        scale_src = weights["hc_scale"]
        self.scale = float((scale_src() if callable(scale_src) else scale_src).flatten().tolist()[0])

    def forward(self, streams: ttnn.Tensor) -> ttnn.Tensor:
        """``streams`` ``[1, T, hc, D]`` TILE -> ``[1, 1, T, D]`` TILE, the collapsed sequence."""
        _, t, hc, d = streams.shape
        if hc != self.hc or d != self.hidden or streams.shape[0] != 1:
            raise ValueError(f"expected streams [1, T, {self.hc}, {self.hidden}], got {tuple(streams.shape)}")

        normed = ttnn.rms_norm(flatten_streams(streams), epsilon=self.norm_eps)
        mixes = self.fn(normed)  # [1, 1, T, hc]
        ttnn.deallocate(normed)
        pre = ttnn.add(ttnn.sigmoid(ttnn.add(ttnn.multiply(mixes, self.scale), self.base)), self.eps)
        ttnn.deallocate(mixes)

        # Weight each stream by its own per-token scalar and sum the streams. Streams go stream-major
        # ([1, hc, T, D]) so the sum runs over a plain outer axis, and ``pre`` to [1, hc, T, 1] so it
        # broadcasts across D: in ``[1, T, hc, D]`` the ``hc`` rows of a token share one tile.
        rows = ttnn.to_layout(streams, ttnn.ROW_MAJOR_LAYOUT)
        stream_major = ttnn.to_layout(ttnn.permute(rows, (0, 2, 1, 3)), ttnn.TILE_LAYOUT)
        ttnn.deallocate(rows)
        weights = ttnn.permute(pre, (0, 3, 2, 1))
        ttnn.deallocate(pre)
        weighted = ttnn.multiply(stream_major, weights)
        ttnn.deallocate(stream_major)
        ttnn.deallocate(weights)
        out = ttnn.sum(weighted, dim=1, keepdim=True)  # [1, 1, T, D]
        ttnn.deallocate(weighted)
        return out


class DeepSeekV4PrefillModel(DeepSeekV4Module):
    """ttnn prefill port of ``DeepseekV4ForCausalLM`` (see the module docstring).

    ``num_layers`` builds only the first layers of the stack (bring-up, or a reduced test model). Weight
    dtypes: ``weight_dtype`` for the attention projections, ``moe_weight_dtype`` for the routers and the
    shared experts, ``expert_dtype`` for the routed experts this class uploads (``None`` takes the system
    profile's default) and ``head_dtype`` for ``lm_head``.

    ``device`` holds the embedding. ``layer_devices`` (one entry per layer, default ``device`` for all)
    places each layer; the head lives with the last layer. ``tp_size`` is the width of the ``1 x tp_size``
    meshes the layers run on (1 for a single device), and ``dense_csa`` lifts the CSA length limit without
    the lightning indexer (see :class:`~.attention.DeepSeekV4PrefillAttention`).
    """

    def __init__(
        self,
        config,
        weights: dict,
        device: ttnn.MeshDevice,
        rope: dict,
        expert_provider: Optional[Callable[[int], Callable[[int], tuple]]] = None,
        experts: Optional[Sequence[DeepSeekV4PreloadedExperts]] = None,
        num_layers: Optional[int] = None,
        cache: Optional[WeightCache] = None,
        weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
        moe_weight_dtype: ttnn.DataType = ttnn.bfloat16,
        expert_dtype: Optional[ttnn.DataType] = None,
        head_dtype: ttnn.DataType = ttnn.bfloat16,
        tp_size: int = 1,
        layer_devices: Optional[Sequence[ttnn.MeshDevice]] = None,
        dense_csa: bool = False,
        progress: Optional[Callable[..., None]] = None,
    ):
        """Upload the embedding, the layers, the head and ``lm_head`` to their devices.

        ``progress(message, important=False)`` is told what the model is about to do, at a fine grain
        (each build step, each layer of each chunk, each stage hand-off, each wait on the device). A caller
        keeps the last message to tell a slow step from a hung one, and logs the ``important`` ones (chunk
        boundaries) always and the rest as it sees fit.
        """
        self._progress = progress or (lambda message, important=False: None)
        self._tag = "build"
        note = self._note
        num_layers = config.num_hidden_layers if num_layers is None else num_layers
        if not 0 < num_layers <= config.num_hidden_layers:
            raise ValueError(f"num_layers {num_layers} is not in [1, {config.num_hidden_layers}]")
        if experts is None and expert_provider is None:
            raise ValueError("pass either expert_provider (to upload the routed experts) or ready-made experts")
        if experts is not None and len(experts) < num_layers:
            raise ValueError(f"got {len(experts)} experts objects for {num_layers} layers")
        if layer_devices is None:
            layer_devices = [device] * num_layers
        if len(layer_devices) != num_layers:
            raise ValueError(f"got {len(layer_devices)} layer devices for {num_layers} layers")
        self.config = config
        self.device = device
        self.layer_devices = list(layer_devices)
        self.head_device = self.layer_devices[-1]
        self.tp_size = tp_size
        self.hidden = config.hidden_size
        self.vocab_size = config.vocab_size
        self.hc = config.hc_mult
        cache = _as_cache(cache)

        note(f"embedding table [{config.vocab_size} x {config.hidden_size}] -> device", important=True)
        embed_file = cache.file("embed_tokens.prefill")
        embed = _materialize(weights["embed_tokens.weight"], embed_file, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        # ``ttnn.embedding`` wants a ROW_MAJOR table.
        self.embedding_weight = _load_weight(
            embed.detach() if embed is not None else None,
            device,
            cache_file_name=embed_file,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

        self.layers: list[DeepSeekV4PrefillDecoderLayer] = []
        for i in range(num_layers):
            layer_device = self.layer_devices[i]
            layer_cache = cache.sub(f"layers.{i}")
            layer_experts = experts[i] if experts is not None else None
            note(f"layer {i + 1}/{num_layers} ({config.layer_types[i]}): routed experts", important=True)
            if layer_experts is None:
                layer_experts = DeepSeekV4PreloadedExperts(
                    config,
                    self._noting_provider(expert_provider(i), i, num_layers, config.num_local_experts),
                    layer_device,
                    dtype=expert_dtype,
                    cache=layer_cache.sub("mlp"),
                    tp_size=tp_size,
                )
            self.layers.append(
                DeepSeekV4PrefillDecoderLayer(
                    config,
                    i,
                    _strip_prefix(weights, f"layers.{i}"),
                    layer_device,
                    rope,
                    experts=layer_experts,
                    cache=layer_cache,
                    weight_dtype=weight_dtype,
                    moe_weight_dtype=moe_weight_dtype,
                    tp_size=tp_size,
                    dense_csa=dense_csa,
                )
            )
            note(f"layer {i + 1}/{num_layers}: built (attention, router, shared expert, hyper-connections)")

        note("final hyper head, norm and lm_head -> device", important=True)
        self.hc_head = DeepSeekV4PrefillHyperHead(
            config, _strip_prefix(weights, "hc_head"), self.head_device, cache=cache.sub("hc_head")
        )
        self.norm_weight = load_norm_gamma(weights["norm.weight"], self.head_device, cache.file("norm.prefill"))
        self.eps = config.rms_norm_eps
        self.lm_head = Linear(
            weights["lm_head.weight"], self.head_device, cache.file("lm_head.prefill"), dtype=head_dtype
        )

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def devices(self) -> list[ttnn.MeshDevice]:
        """The distinct devices the model uses, in pipeline order."""
        seen: dict[int, ttnn.MeshDevice] = {}
        for dev in [self.device, *self.layer_devices, self.head_device]:
            seen.setdefault(id(dev), dev)
        return list(seen.values())

    def new_state(self) -> list[PrefillAttentionState]:
        """One empty attention state per layer: the start of a prompt."""
        return [layer.new_state() for layer in self.layers]

    def synchronize(self, why: str = "") -> None:
        """Block until every device the model uses has finished its queued work."""
        devices = self.devices
        for i, dev in enumerate(devices):
            self._note(f"{self._tag}: {why or 'synchronize'} - waiting for device {i + 1}/{len(devices)}")
            ttnn.synchronize_device(dev)

    # ------------------------------------------------------------------ progress
    def _note(self, message: str, important: bool = False) -> None:
        """Report what the model is doing (see the constructor's ``progress``)."""
        self._progress(message, important)

    def _noting_provider(self, provider: Callable[[int], tuple], layer: int, num_layers: int, num_experts: int):
        """``provider`` that reports every 32nd expert it is asked for (only a cache miss calls it at all)."""

        def noting(e: int):
            if e % 32 == 0:
                self._note(
                    f"layer {layer + 1}/{num_layers}: reading + quantizing routed expert {e}/{num_experts} "
                    "from the checkpoint (weight-cache miss)"
                )
            return provider(e)

        return noting

    # ------------------------------------------------------------------ pieces
    def _host_ids(self, input_ids) -> torch.Tensor:
        """``input_ids`` (``[T]`` / ``[1, T]`` ints) as a validated ``[1, T]`` int64 torch tensor."""
        ids = torch.as_tensor(input_ids)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        if ids.dim() != 2 or ids.shape[0] != 1:
            raise ValueError(f"expected token ids [T] or [1, T], got {tuple(ids.shape)}")
        if ids.numel() == 0 or ids.shape[1] % ALIGNMENT:
            raise ValueError(f"prompt length {ids.shape[1]} must be a positive multiple of {ALIGNMENT}")
        if int(ids.min()) < 0 or int(ids.max()) >= self.vocab_size:
            raise ValueError(f"token ids must lie in [0, {self.vocab_size})")
        return ids.long()

    def _upload_ids(self, ids: torch.Tensor, device: Optional[ttnn.MeshDevice] = None) -> ttnn.Tensor:
        """``[1, T]`` ids as the uint32 ROW_MAJOR tensor the embedding and hash routers read (on every rank)."""
        device = self.device if device is None else device
        return ttnn.from_torch(
            ids.to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if device.get_num_devices() > 1 else None,
        )

    @staticmethod
    def to_host(tensor: ttnn.Tensor, device: ttnn.MeshDevice) -> torch.Tensor:
        """One rank's copy of a replicated device tensor, as fp32 torch (the tensors here are replicated)."""
        if device.get_num_devices() > 1:
            tensor = ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
            return tensor[: tensor.shape[0] // device.get_num_devices()].to(torch.float32)
        return ttnn.to_torch(tensor).to(torch.float32)

    def _handoff(self, streams: ttnn.Tensor, src: ttnn.MeshDevice, dst: ttnn.MeshDevice) -> ttnn.Tensor:
        """Move the residual streams ``[1, T, hc, D]`` to the next pipeline stage, through the host."""
        host = self.to_host(streams, src).to(torch.bfloat16)
        return ttnn.from_torch(
            host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dst,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(dst) if dst.get_num_devices() > 1 else None,
        )

    def embed(self, ids_dev: ttnn.Tensor) -> ttnn.Tensor:
        """``[1, T]`` uint32 ids -> the initial streams ``[1, T, hc, D]`` TILE: the embedding, once per stream."""
        t = ids_dev.shape[1]
        emb = ttnn.embedding(ids_dev, self.embedding_weight, layout=ttnn.ROW_MAJOR_LAYOUT)  # [1, T, D]
        # ROW_MAJOR, where adding the stream axis is a view and repeating along it a page copy.
        emb = ttnn.reshape(emb, [1, t, 1, self.hidden])
        streams = ttnn.repeat(emb, ttnn.Shape([1, 1, self.hc, 1]))
        return ttnn.to_layout(streams, ttnn.TILE_LAYOUT)

    def head(self, streams: ttnn.Tensor, last_only: bool = True) -> ttnn.Tensor:
        """``streams`` ``[1, T, hc, D]`` -> logits ``[1, 1, 1, V]`` (the last token) or ``[1, 1, T, V]``."""
        self._note(f"{self._tag}: head (hyper head, final norm, lm_head)")
        hidden = ttnn.rms_norm(self.hc_head(streams), weight=self.norm_weight, epsilon=self.eps)  # [1, 1, T, D]
        if last_only:
            t = hidden.shape[2]
            hidden = ttnn.slice(hidden, [0, 0, t - 1, 0], [1, 1, t, self.hidden])
        return self.lm_head(hidden)

    def _stack(
        self,
        ids: torch.Tensor,
        states: list[PrefillAttentionState],
        on_layer: Optional[Callable[[int, ttnn.Tensor, ttnn.MeshDevice], None]] = None,
    ) -> ttnn.Tensor:
        """The embedding and every layer over one chunk of ``ids`` ``[1, T]``; the final streams (on the head device).

        ``on_layer(index, streams, device)`` (verification) is called with each layer's output streams
        before they move on; it must not deallocate them.
        """
        if len(states) != len(self.layers):
            raise ValueError(f"expected {len(self.layers)} layer states, got {len(states)}")
        ids_on: dict[int, ttnn.Tensor] = {}

        def ids_for(dev: ttnn.MeshDevice) -> ttnn.Tensor:
            if id(dev) not in ids_on:
                ids_on[id(dev)] = self._upload_ids(ids, dev)
            return ids_on[id(dev)]

        self._note(f"{self._tag}: embedding")
        streams = self.embed(ids_for(self.device))
        current = self.device
        stage = 0
        for i, (layer, dev, state) in enumerate(zip(self.layers, self.layer_devices, states)):
            if dev is not current:
                stage += 1
                self._note(f"{self._tag}: stage hand-off to stage {stage} (streams -> host -> next submesh)")
                moved = self._handoff(streams, current, dev)
                ttnn.deallocate(streams)
                streams, current = moved, dev
                self._note(f"{self._tag}: stage hand-off done")
            kind = self.config.layer_types[i].replace("_attention", "")
            self._note(f"{self._tag}: layer {i + 1}/{len(self.layers)} ({kind}, stage {stage}) enqueue")
            out = layer(streams, state, ids_for(dev))
            if on_layer is not None:
                on_layer(i, out, dev)
            ttnn.deallocate(streams)
            streams = out
        for t in ids_on.values():
            ttnn.deallocate(t)
        return streams

    # ------------------------------------------------------------------ public
    def forward(
        self, input_ids, states: Optional[list[PrefillAttentionState]] = None, last_only: bool = True
    ) -> ttnn.Tensor:
        """The network over one chunk of a prompt.

        ``input_ids`` is ``[T]`` / ``[1, T]`` torch ints with ``T`` a multiple of ``ALIGNMENT``;
        ``states`` is this model's per-layer state (:meth:`new_state`), advanced in place -- ``None`` runs a
        whole prompt from scratch and discards it. Returns bf16 TILE logits ``[1, 1, 1, V]`` for the chunk's
        last token, or ``[1, 1, T, V]`` for every token with ``last_only=False``, on the head device
        (replicated across its ranks; read one with :meth:`to_host`).
        """
        ids = self._host_ids(input_ids)
        self._tag = "forward"
        streams = self._stack(ids, self.new_state() if states is None else states)
        logits = self.head(streams, last_only)
        ttnn.deallocate(streams)
        return logits

    def prefill(
        self,
        input_ids,
        chunk_size: int = 1024,
        on_chunk: Optional[Callable[[int, int, int, float], None]] = None,
        states: Optional[list[PrefillAttentionState]] = None,
    ) -> tuple[ttnn.Tensor, list[PrefillAttentionState]]:
        """A whole prompt as consecutive ``chunk_size`` chunks; returns ``(last-token logits, states)``.

        The logits are ``[1, 1, 1, V]`` for the prompt's last token, and ``states`` is what decode (or a
        further chunk) continues from. Only the last chunk pays for the head.

        ``on_chunk(index, start, end, seconds)`` is called after every chunk with its wall time. Giving it
        makes the model synchronize every device before and after each chunk, so ``seconds`` covers all of
        the chunk's device work (the last chunk includes the head); without it the chunks are enqueued back
        to back. ``states`` continues an earlier prefill instead of starting a new one.
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
            streams = self._stack(ids[:, start:end], states)
            last = end == total
            logits = self.head(streams, last_only=True) if last else None
            ttnn.deallocate(streams)
            self._note(f"{self._tag}: all work enqueued in {time.perf_counter() - t0:.2f} s")
            if on_chunk is not None:
                self.synchronize("device compute")
                on_chunk(index, start, end, time.perf_counter() - t0)
        return logits, states
