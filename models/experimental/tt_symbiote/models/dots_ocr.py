# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Standalone TTNN inference pipeline for dots.ocr.

Replaces HF model.generate() entirely, combining scatter-merge on device,
argmax on device, and a custom generation loop.  No monkey-patching, no
HF generate dependency.

Logits are cast to float32 before greedy argmax, and the LM head uses FP32 destination
accumulation, for stable token choices across mesh layouts (same cost as the prior fast
path in practice for this pipeline).
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch
import ttnn
from ttnn.device import is_wormhole_b0


from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.core.run_config import TracedRun, disable_trace, trace_enabled
from models.experimental.tt_symbiote.modules.attention import (
    PagedAttentionConfig,
    TTNNPagedAttentionKVCache,
    dp_batch_shard_tensor_mapper,
)
from models.experimental.tt_symbiote.modules.dots_ocr_decoder_layer import (
    TTNNDotsOCRDecoderLayer,
    TTNNDotsOCRLayerStack,
    TTNNDotsOCRLocalShardRMSNorm,
)
from models.experimental.tt_symbiote.modules.dots_ocr_vision import TTNNDotsOCRVisionTower
from models.experimental.tt_symbiote.modules.embedding import TTNNEmbedding
from models.experimental.tt_symbiote.modules.linear import (
    TTNNDotsOCRDRAMShardedLMHead,
)
from models.experimental.tt_symbiote.modules.normalization import TTNNDistributedRMSNorm
from models.experimental.tt_symbiote.utils.device_management import timed_call


# Mixed-precision decode gate_up: number of leading decoder layers whose fused
# gate_up projection keeps the fast BFP4 weight. Layers at/after this index use
# BFP8 (more precise, ~2x the DRAM-sharded weight read). See _build_symbiote_stack.
DOTS_OCR_GATE_UP_BF4_LAYERS = int(os.environ.get("DOTS_OCR_GATE_UP_BF4_LAYERS", "7"))


def _argmax_token_on_device(logits: ttnn.Tensor, use_multicore: bool = True) -> ttnn.Tensor:
    """Greedy token from logits.

    Note on the reduce-axis-sharding rule: literally that rule says never
    split the reduce axis across cores. For ``argmax(dim=-1)`` on
    ``[1, 1, vocab=152064]`` the non-reduce extent is 1, so the rule
    points at single-core. We tried ``use_multicore=False`` and it
    regressed wall-clock by ~3 ms/iter (decode 3.62 s -> 4.17 s at DP=8).
    The single-core kernel becomes a sequential 152064-element scan with
    a small read pipeline, while the multi-core kernel parallelizes the
    scan and pays a small NoC fold at the end -- for vocab this large the
    fold cost is dwarfed by the parallel speedup. Stay on
    ``use_multicore=True`` here; the rule still applies as written for
    "normal" reduction shapes (small reduce dim, larger non-reduce
    extent), just not for argmax over a giant vocab with M=1.
    """
    logits_rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
    token = ttnn.argmax(
        logits_rm,
        dim=-1,
        keepdim=True,
        use_multicore=use_multicore,
    )
    b = int(token.shape[0])
    return ttnn.reshape(token, (b, 1))


def _unwrap_ttnn_tensor(tt_hidden):
    if hasattr(tt_hidden, "ttnn_tensor") and tt_hidden.ttnn_tensor is not None:
        return tt_hidden.ttnn_tensor
    return tt_hidden


def _sync_profile_enabled() -> bool:
    return os.environ.get("DOTS_OCR_PROFILE_SYNC", "").lower() in {"1", "true", "yes", "on"}


def _defer_decode_readback_enabled() -> bool:
    return os.environ.get("DOTS_OCR_DEFER_DECODE_READBACK", "0").lower() in {"1", "true", "yes", "on"}


def _pipeline_decode_readback_enabled() -> bool:
    return os.environ.get("DOTS_OCR_PIPELINE_DECODE_READBACK", "0").lower() in {"1", "true", "yes", "on"}


def _deep_sync_profile_enabled() -> bool:
    return os.environ.get("DOTS_OCR_PROFILE_DECODE_GRAPH", "").lower() in {"1", "true", "yes", "on"}


def _decode_tp_scheme_from_env() -> str:
    """Tensor-parallel decode scheme for full dots.ocr pipeline construction.

    ``col_parallel`` is the default fast TP4 decode path. Set
    ``DOTS_OCR_TP_DECODE_SCHEME=row`` explicitly to compare the older
    row-parallel path.
    """
    scheme = os.environ.get("DOTS_OCR_TP_DECODE_SCHEME", "col_parallel").strip()
    if scheme not in {"row", "col_parallel"}:
        raise ValueError("DOTS_OCR_TP_DECODE_SCHEME must be one of {'row', 'col_parallel'}, " f"got {scheme!r}")
    return scheme


def _default_text_body(device) -> str:
    if is_wormhole_b0() and _tp_degree(device) == 4:
        return "tp4_prefill"
    return "symbiote"


def _text_body_from_env(device=None) -> str:
    """Which text-decoder body the pipeline builds.

    ``symbiote`` is the existing hidden-sharded ``TTNNDotsOCRLayerStack``
    (``row`` / ``col_parallel`` schemes). ``tp4`` swaps in the head-local
    replicated-hidden Megatron body from ``models.experimental.dots_ocr_tp4`` --
    its prefill is ~4x lighter on SDPA (3 Q-heads / 1 KV-head per chip, no
    QKV gather) so the prefill is much faster. Both prefill and decode run on
    that body (they must share one KV-cache layout: ``num_kv_heads=1`` per chip),
    but the per-token decode machinery (traced graph, on-device argmax, on-device
    token feedback) is the unchanged symbiote pipeline path.

    The TP4 body consumes a *replicated full-H* hidden stream; the existing
    embedding/vision/scatter still produce the ``H/num_devices`` column shard, so
    the graphs all-gather to full-H at the body boundary (one CCL, see
    ``_all_gather_hidden_to_full``).
    ``tp4_prefill`` uses the TP4 body only for prefill, fills a decode-compatible
    two-KV-head cache, and keeps the existing symbiote decode stack. On
    Wormhole TP4 meshes this is the default optimized prefill path; set
    ``DOTS_OCR_TEXT_BODY=symbiote`` to force the legacy prefill body.
    """
    body_env = os.environ.get("DOTS_OCR_TEXT_BODY")
    body = body_env.strip().lower() if body_env is not None else _default_text_body(device)
    if body not in {"symbiote", "tp4", "tp4_prefill"}:
        raise ValueError(f"DOTS_OCR_TEXT_BODY must be one of {{'symbiote', 'tp4', 'tp4_prefill'}}, got {body!r}")
    return body


def _tp4_prefill_body_enabled(device=None) -> bool:
    return _text_body_from_env(device) == "tp4_prefill"


def _tp_cluster_axis(device) -> int:
    """Mesh axis the TP shards live on (the >1 dim of a 1xN / Nx1 mesh)."""
    shape = [int(x) for x in device.shape]
    return 0 if (len(shape) == 2 and shape[0] > 1 and shape[1] == 1) else 1


def _mesh_shape_2d(device) -> tuple[int, int] | None:
    if not hasattr(device, "shape"):
        return None
    shape = tuple(int(x) for x in device.shape)
    return shape if len(shape) == 2 else None


def _tp_degree(device) -> int:
    shape = _mesh_shape_2d(device)
    return int(shape[1]) if shape is not None else 1


def _dp_degree(device) -> int:
    shape = _mesh_shape_2d(device)
    if shape is None:
        return 1
    return int(shape[0]) if int(shape[0]) > 1 else 1


def _dp_batch_mesh_composer(device, batch_size: int):
    """Compose small DP-sharded tensors; callers drop replicated TP copies."""
    shape = _mesh_shape_2d(device)
    if shape is None or int(batch_size) <= 1:
        return None
    if int(shape[0]) == int(batch_size):
        return ttnn.ConcatMesh2dToTensor(device, shape, (0, 1))
    if int(shape[1]) == int(batch_size) and int(shape[0]) == 1:
        return ttnn.ConcatMesh2dToTensor(device, shape, (1, 0))
    return None


def _token_mesh_composer(device, batch_size: int, dual: bool):
    shape = _mesh_shape_2d(device)
    if shape is not None:
        if dual:
            return _dp_batch_mesh_composer(device, batch_size)
        if int(shape[0]) == 1 and int(shape[1]) > 1:
            return ttnn.ConcatMesh2dToTensor(device, shape, (1, 0))
        if int(shape[1]) == 1 and int(shape[0]) > 1:
            return ttnn.ConcatMeshToTensor(device, dim=0)
        return ttnn.ConcatMesh2dToTensor(device, shape, (0, 1))
    return ttnn.ConcatMeshToTensor(device, dim=0)


def _select_dp_token_replicas(token_torch: torch.Tensor, device, batch_size: int) -> torch.Tensor:
    """Keep one TP replica per DP stream after 2D token readback."""
    shape = _mesh_shape_2d(device)
    if shape is None or int(batch_size) <= 1 or int(shape[0]) != int(batch_size) or int(shape[1]) <= 1:
        return token_torch
    elems_per_stream = int(token_torch.numel()) // (int(batch_size) * int(shape[1]))
    if elems_per_stream <= 0 or int(token_torch.numel()) % (int(batch_size) * int(shape[1])) != 0:
        return token_torch
    return token_torch.reshape(int(batch_size), int(shape[1]), elems_per_stream)[:, 0, :]


def _first_device_token_to_torch(token_tt: ttnn.Tensor) -> torch.Tensor:
    shards = ttnn.get_device_tensors(token_tt)
    if not shards:
        return ttnn.to_torch(token_tt)
    return ttnn.to_torch(shards[0])


def _all_gather_hidden_to_full(t: ttnn.Tensor, device, hidden_size: int) -> ttnn.Tensor:
    """All-gather an ``[.., H/ndev]`` column-sharded hidden -> replicated ``[.., H]``.

    Bridges the symbiote column-sharded hidden stream (embedding / vision scatter
    emit ``H/num_devices`` per chip) to the replicated full-H stream the TP4 body
    consumes. No-op when the tensor is already full-width or the mesh is single
    device. Mirrors ``dots_ocr_tp4.tt.common.all_gather_last_dim`` (plain
    ``ttnn.all_gather``, no worker kwargs -- proven trace-capturable in the TP4
    graphs).
    """
    if int(t.shape[-1]) >= int(hidden_size):
        return t
    if _tp_degree(device) <= 1:
        return t
    orig_shape = list(t.shape)
    work = t if len(orig_shape) == 4 else ttnn.reshape(t, [1] * (4 - len(orig_shape)) + orig_shape)
    gathered = ttnn.all_gather(
        work,
        dim=3,
        num_links=1,
        cluster_axis=_tp_cluster_axis(device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Linear,
    )
    if len(orig_shape) < 4:
        gathered = ttnn.reshape(gathered, orig_shape[:-1] + [int(gathered.shape[-1])])
    return gathered


def _dots_ocr_signpost(header: str) -> None:
    """Emit a Tracy ``TT_SIGNPOST`` marker (visible in device op logs / perf reports)."""
    try:
        from tools.tracy import signpost
    except ImportError:
        return
    signpost(header)


@contextlib.contextmanager
def _profile_stage(device, name: str):
    if not _sync_profile_enabled():
        yield
        return
    ttnn.synchronize_device(device)
    start = time.time()
    try:
        yield
    finally:
        ttnn.synchronize_device(device)
        elapsed_ms = (time.time() - start) * 1000.0
        print(f"[DOTS_OCR_PROFILE_SYNC] {name}: {elapsed_ms:.3f} ms")


@contextlib.contextmanager
def _profile_graph_stage(device, name: str):
    if not _deep_sync_profile_enabled():
        yield
        return
    ttnn.synchronize_device(device)
    start = time.time()
    try:
        yield
    finally:
        ttnn.synchronize_device(device)
        elapsed_ms = (time.time() - start) * 1000.0
        print(f"[DOTS_OCR_PROFILE_SYNC] {name}: {elapsed_ms:.3f} ms")


def _dp_repack_batch_sharded_hidden_for_device(device, batch_input_mapper, tt_hidden: ttnn.Tensor) -> ttnn.Tensor:
    """Re-materialize a DP batch-sharded hidden tensor with the pipeline batch mapper.

    Some TTNN ops preserve a global logical batch shape even though DP owns one
    row per device. A host compose + re-upload makes the local shard layout
    match the token-id/embedding mapper before traced decoder execution.
    """
    if batch_input_mapper is None:
        return tt_hidden

    t = _unwrap_ttnn_tensor(tt_hidden)
    ttnn.synchronize_device(device)
    owned = ttnn.clone(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    mesh_shape = tuple(int(x) for x in device.shape)
    full = ttnn.to_torch(
        owned,
        mesh_composer=ttnn.ConcatMesh2dToTensor(device, mesh_shape, (0, -1)),
    )
    ttnn.deallocate(owned)
    hidden_mapper = batch_input_mapper
    if len(mesh_shape) == 2 and int(mesh_shape[0]) == int(full.shape[0]):
        hidden_mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=mesh_shape, dims=(0, -1))
    return ttnn.from_torch(
        full,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=hidden_mapper,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _dp_readback_batch_size() -> int:
    raw = os.environ.get("DOTS_OCR_DP_READBACK_BATCH", "1").strip().lower()
    try:
        k = int(raw)
    except ValueError:
        k = 1
    return max(1, min(k, 32))


def _normalize_image_grid_thw_torch(grid: torch.Tensor) -> torch.Tensor:
    """Return CPU ``[N, 3]`` long (t, h, w) for vision RoPE.

    Passed as ``mm_grid_thw=...`` into ``TTNNDotsOCRPrefillGraph`` so trace capture never
    reads ``tt_grid`` from device via ``ttnn.to_torch``.
    """
    g = grid.detach().to(torch.long)
    if g.dim() == 1:
        if int(g.numel()) != 3:
            raise ValueError(f"image_grid_thw must be length-3 or [N,3], got shape {tuple(g.shape)}")
        g = g.unsqueeze(0)
    elif g.dim() > 2:
        g = g.reshape(-1, int(g.shape[-1]))
    if g.dim() != 2 or int(g.shape[1]) != 3:
        raise ValueError(f"grid_thw must be [N,3], got {tuple(g.shape)}")
    if int(g.shape[0]) > 1 and bool(torch.all(g == g[0], dim=1).all().item()):
        g = g[:1]
    return g


@trace_enabled
class TTNNDotsOCRPrefillGraph(TTNNModule):
    def preprocess_weights_impl(self):
        return self

    def move_weights_to_device_impl(self):
        return self

    def __init__(
        self,
        decoder_stack,
        final_norm,
        lm_head,
        embedding=None,
        vision_tower=None,
        image_token_id: int = 151665,
        hidden_size: int = 1536,
        scatter_uses_dp_batch_mapper: bool = False,
        gather_hidden_for_full_body: bool = False,
    ):
        super().__init__()
        self._p_stack = decoder_stack
        self._p_norm = final_norm
        self._p_lm = lm_head
        self._p_embedding = embedding
        self._p_vision = vision_tower
        self._image_token_id = int(image_token_id)
        self._hidden_size = int(hidden_size)
        self._scatter_uses_dp_batch_mapper = bool(scatter_uses_dp_batch_mapper)
        # TP4 body consumes replicated full-H; gather the column-sharded hidden
        # to full width at the body boundary (see _all_gather_hidden_to_full).
        self._gather_hidden_for_full_body = bool(gather_hidden_for_full_body)
        self._scatter_cache_input_ids: Optional[torch.Tensor] = None
        self._scatter_cache_key: Optional[tuple] = None
        self._scatter_cache_idx: Optional[ttnn.Tensor] = None
        self._scatter_cache_mask: Optional[ttnn.Tensor] = None
        self._scatter_zero_row_key: Optional[tuple] = None
        self._scatter_zero_row: Optional[ttnn.Tensor] = None

    def release_scatter_cache(self) -> None:
        if self._scatter_cache_idx is not None:
            ttnn.deallocate(self._scatter_cache_idx)
            self._scatter_cache_idx = None
        if self._scatter_cache_mask is not None:
            ttnn.deallocate(self._scatter_cache_mask)
            self._scatter_cache_mask = None
        if self._scatter_zero_row is not None:
            ttnn.deallocate(self._scatter_zero_row)
            self._scatter_zero_row = None
            self._scatter_zero_row_key = None
        self._scatter_cache_input_ids = None
        self._scatter_cache_key = None

    def get_or_build_scatter_tensors(
        self,
        input_ids: torch.Tensor,
        n_vision: int,
        idx_mapper,
        tp_degree: int,
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        device = self.device
        S = int(input_ids.shape[-1])
        H = self._hidden_size
        H_per_device = H // tp_degree if tp_degree > 1 else H
        cache_key = (
            int(input_ids.shape[0]),
            S,
            int(n_vision),
            int(H_per_device),
            self._scatter_uses_dp_batch_mapper,
        )
        cache_hit = (
            self._scatter_cache_key == cache_key
            and self._scatter_cache_input_ids is not None
            and torch.equal(self._scatter_cache_input_ids, input_ids)
            and self._scatter_cache_idx is not None
            and self._scatter_cache_mask is not None
        )
        if cache_hit:
            return self._scatter_cache_idx, self._scatter_cache_mask

        gather_idx = torch.zeros(int(input_ids.shape[0]), S, dtype=torch.int32)
        for b in range(int(input_ids.shape[0])):
            img_mask_b = input_ids[b] == self._image_token_id
            img_positions = img_mask_b.nonzero(as_tuple=True)[0]
            n_img = min(len(img_positions), int(n_vision))
            gather_idx[b, img_positions[:n_img]] = torch.arange(1, n_img + 1, dtype=torch.int32)

        tt_idx = ttnn.from_torch(
            gather_idx,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=idx_mapper,
        )

        img_mask = input_ids == self._image_token_id
        mask_float = img_mask.float().unsqueeze(-1)
        tt_mask = ttnn.from_torch(
            mask_float,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=idx_mapper,
        )
        if self._scatter_cache_idx is not None:
            ttnn.deallocate(self._scatter_cache_idx)
        if self._scatter_cache_mask is not None:
            ttnn.deallocate(self._scatter_cache_mask)
        self._scatter_cache_key = cache_key
        self._scatter_cache_input_ids = input_ids.detach().clone()
        self._scatter_cache_idx = tt_idx
        self._scatter_cache_mask = tt_mask
        return tt_idx, tt_mask

    def _ensure_scatter_zero_row(self, tp_degree: int, H: int, H_per_device: int) -> None:
        device = self.device
        if tp_degree > 1:
            zero_row_mapper = ttnn.ShardTensor2dMesh(
                device,
                dims=(None, -1),
                mesh_shape=list(device.shape),
            )
        else:
            zero_row_mapper = ttnn.ReplicateTensorToMesh(device)

        zero_row_key = (int(H), int(tp_degree), self._scatter_uses_dp_batch_mapper)
        if self._scatter_zero_row is None or self._scatter_zero_row_key != zero_row_key:
            if self._scatter_zero_row is not None:
                ttnn.deallocate(self._scatter_zero_row)
            self._scatter_zero_row = ttnn.from_torch(
                torch.zeros(1, H, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=zero_row_mapper,
            )
            self._scatter_zero_row_key = zero_row_key

    def _scatter_fuse_text_and_vision(
        self,
        text_embeds: ttnn.Tensor,
        vision_tt: ttnn.Tensor,
        tt_idx: ttnn.Tensor,
        tt_mask: ttnn.Tensor,
    ) -> ttnn.Tensor:
        device = self.device
        tp_degree = _tp_degree(device)
        N_vision = int(vision_tt.shape[2])
        H_per_device = int(vision_tt.shape[3])
        scatter_memory_config = ttnn.L1_MEMORY_CONFIG

        vision_2d = ttnn.reshape(vision_tt, (N_vision, H_per_device))

        self._ensure_scatter_zero_row(tp_degree, self._hidden_size, H_per_device)
        vision_table = ttnn.concat([self._scatter_zero_row, vision_2d], dim=0, memory_config=scatter_memory_config)
        # Do NOT deallocate vision_tt / vision_2d here: vision_tt is now a
        # persistent trace-input buffer (vision runs eager outside the trace and
        # its output is copied into this buffer on every replay), and vision_2d
        # is a reshape view that shares its storage. Freeing either would corrupt
        # replay. The eager caller drops its ref so GC frees it after prefill.

        full_vision_col_sharded = ttnn.embedding(
            tt_idx,
            vision_table,
            layout=ttnn.TILE_LAYOUT,
            memory_config=scatter_memory_config,
        )
        ttnn.deallocate(vision_table)

        fused = ttnn.where(tt_mask, full_vision_col_sharded, text_embeds, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(full_vision_col_sharded)
        return fused

    def forward(
        self, hidden_states, cache_position, *mm_args, past_key_value=None, mm_grid_thw: Optional[torch.Tensor] = None
    ):
        h0 = hidden_states
        if (
            self._p_embedding is not None
            and isinstance(h0, ttnn.Tensor)
            and h0.dtype
            in (
                ttnn.uint32,
                ttnn.int32,
            )
        ):
            text_e = self._p_embedding.forward(h0)
        else:
            text_e = h0

        if len(mm_args) == 3:
            # The vision trunk already ran EAGER in ``prefill`` (it does host
            # reads and has large CBs that are illegal / clash inside trace
            # capture). We receive its precomputed output ``vision_tt`` as a
            # trace input and only fuse it into the text embeds here -- this
            # whole forward is then trace-safe. ``vision_tt`` is a persistent
            # trace-input buffer, so ``_scatter_fuse_text_and_vision`` must NOT
            # deallocate it (its contents are re-copied on every replay).
            vision_tt, tt_idx, tt_mask = mm_args
            text_e = self._scatter_fuse_text_and_vision(text_e, vision_tt, tt_idx, tt_mask)

        if self._gather_hidden_for_full_body:
            # No-op when the hidden is already full-width (the multimodal dual path
            # scatter-fuses + gathers EAGER and passes a full-H hidden, so this only
            # gathers for the in-graph (non-prefused) paths). The plain-CCL all_gather
            # does per-device host writes (semaphore setup) that are illegal during
            # trace capture, which is why the prefused path keeps it out of the trace.
            text_e = _all_gather_hidden_to_full(text_e, self.device, self._hidden_size)

        h = self._p_stack.forward(text_e, past_key_value=past_key_value, cache_position=cache_position)
        if self._gather_hidden_for_full_body:
            # TP4 path: return the post-body hidden; final norm + lm_head + argmax
            # run EAGER outside this captured trace. The in-trace full-vocab
            # all_gather + argmax corrupts its output buffer when sharing a trace
            # with the heavy TP4 body (proven: eager works, traced does not).
            # Slice the real last token now (trace-safe) so the eager tail only
            # processes [B, 1, H], and its tile-broadcast RMSNorm gamma is valid.
            sl = int(h.shape[-2])
            if sl > 1:
                b = int(h.shape[0])
                hd = int(h.shape[-1])
                h = ttnn.slice(h, [0, sl - 1, 0], [b, sl, hd])
            return h
        h = self._p_norm.forward(h)
        sl = int(h.shape[-2])
        if sl > 1:
            b = int(h.shape[0])
            hd = int(h.shape[-1])
            h = ttnn.slice(h, [0, sl - 1, 0], [b, sl, hd])
        logits = self._p_lm.forward(h)
        return _argmax_token_on_device(logits)

    def post_trace_execute(self, func_args, func_kwargs, result):
        past_key_value = func_kwargs.get("past_key_value")
        if past_key_value is None or not hasattr(past_key_value, "update_seq_length"):
            return
        hid = func_args[0]
        if isinstance(hid, ttnn.Tensor) and hid.dtype in (ttnn.uint32, ttnn.int32):
            seq_len = int(hid.shape[-1])
        elif isinstance(hid, ttnn.Tensor):
            seq_len = int(hid.shape[-2])
        else:
            hid = _unwrap_ttnn_tensor(hid)
            seq_len = int(hid.shape[-2])
        for layer in self._p_stack.layers:
            past_key_value.update_seq_length(layer_idx=layer.self_attn.layer_idx, seq_len=seq_len)


@trace_enabled
class TTNNDotsOCRDecodeGraph(TTNNModule):
    def preprocess_weights_impl(self):
        return self

    def move_weights_to_device_impl(self):
        return self

    def __init__(
        self, decoder_stack, final_norm, lm_head, embedding=None, gather_hidden_for_full_body=False, hidden_size=1536
    ):
        super().__init__()
        self._d_stack = decoder_stack
        self._d_norm = final_norm
        self._d_lm = lm_head
        self._d_embedding = embedding
        # TP4 body consumes replicated full-H; gather the column-sharded
        # per-token embedding to full width at the body boundary.
        self._gather_hidden_for_full_body = bool(gather_hidden_for_full_body)
        self._hidden_size = int(hidden_size)

    def _embed_and_gather(self, decode_input):
        hidden_states = self._d_embedding.forward(decode_input) if self._d_embedding is not None else decode_input
        if self._gather_hidden_for_full_body:
            hidden_states = _all_gather_hidden_to_full(hidden_states, self.device, self._hidden_size)
        return hidden_states

    def forward(self, decode_input, cache_position, past_key_value):
        dev = self.device
        if _deep_sync_profile_enabled():
            with _profile_graph_stage(dev, "decode.graph.embedding"):
                hidden_states = self._embed_and_gather(decode_input)
            with _profile_graph_stage(dev, "decode.graph.layer_stack"):
                h = self._d_stack.forward(hidden_states, past_key_value=past_key_value, cache_position=cache_position)
            if self._gather_hidden_for_full_body:
                # TP4 path: norm + lm_head + argmax run eager outside the trace.
                return h
            with _profile_graph_stage(dev, "decode.graph.final_norm"):
                h = self._d_norm.forward(h)
            with _profile_graph_stage(dev, "decode.graph.lm_head"):
                logits = self._d_lm.forward(h)
            with _profile_graph_stage(dev, "decode.graph.argmax"):
                return _argmax_token_on_device(logits)

        # Default: one straight-line forward when deep graph profiling is off.
        hidden_states = self._embed_and_gather(decode_input)
        h = self._d_stack.forward(hidden_states, past_key_value=past_key_value, cache_position=cache_position)
        if self._gather_hidden_for_full_body:
            # TP4 path: norm + lm_head + argmax run eager outside the trace
            # (see TTNNDotsOCRPrefillGraph for why).
            return h
        h = self._d_norm.forward(h)
        logits = self._d_lm.forward(h)
        return _argmax_token_on_device(logits)

    def post_trace_execute(self, func_args, func_kwargs, result):
        past_key_value = func_kwargs.get("past_key_value")
        if past_key_value is None or not hasattr(past_key_value, "update_seq_length"):
            return
        seq_len = 1
        for layer in self._d_stack.layers:
            past_key_value.update_seq_length(layer_idx=layer.self_attn.layer_idx, seq_len=seq_len)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Configuration for TTNNDotsOCRPipeline."""

    vocab_size: int = 151936
    hidden_size: int = 1536
    image_token_id: int = 151665
    eos_token_ids: List[int] = field(default_factory=lambda: [151643, 151673])
    max_new_tokens: int = 512
    num_devices: int = 2  # N300
    batch_size: int = 1


# ---------------------------------------------------------------------------
# Helper: create paged KV cache (mirrors test helper)
# ---------------------------------------------------------------------------


def _create_paged_kv_cache(model_config, device, batch_size: int = 1):
    """Create a paged attention KV cache for dots.ocr.

    Cache dtype is left at default ``bfloat16``. ``bfloat8_b`` was tried
    twice in isolation -- both runs produced visibly corrupted text
    (``EX丝滿º衔task放...``). The K/V values for dots.ocr decode SDPA
    are sensitive to per-element quantization in a way that is not
    captured by simple per-tile statistics; the BFP8 shared exponent
    appears to be too coarse for the long-horizon attention scores in
    this model. Do not flip back to ``bfloat8_b`` without a per-layer
    Q/K/V max-error sweep.
    """
    head_dim = getattr(
        model_config,
        "head_dim",
        model_config.hidden_size // model_config.num_attention_heads,
    )
    # Keep at least 64 pages per DP stream. The vision prompt is ~2.8K tokens;
    # a fixed 256 global block pool gives only 2K tokens/stream at batch 8.
    blocks_per_sequence = 64
    config = PagedAttentionConfig(
        block_size=64,
        max_num_blocks=max(256, batch_size * blocks_per_sequence),
        batch_size=batch_size,
    )
    return TTNNPagedAttentionKVCache(
        num_layers=model_config.num_hidden_layers,
        num_kv_heads=model_config.num_key_value_heads,
        head_dim=head_dim,
        config=config,
        device=None,
        # KV cache stays BF16. A BFP8/BFP4 cache would ~halve the K/V DRAM read that
        # bounds paged_sdpa_decode (sweep 2026-06-15: BFP8 ~-40%, would hit <8 us), and
        # the SDPA read + cache alloc accept it -- BUT paged_update_cache only takes
        # BF16/FP32 input and does NOT convert it to a BFP8 cache (it writes BF16 bytes
        # into BFP8 tile slots -> garbled output, confirmed in-model for both all-BFP8
        # and V-only-BFP8). So a sub-BF16 cache needs a kernel that converts-on-write
        # (or a BFP8-input update); not a config change. Keep BF16.
    ).to_device(device)


class _TP4PrefillDecodeCacheAdapter(TTNNPagedAttentionKVCache):
    """Fill TP4 prefill cache and symbiote decode cache from TP4 K/V states.

    TP4 attention computes one KV head per chip. The existing fast
    ``col_parallel`` decode path expects both KV heads on each chip. During
    prefill, gather the per-chip TP4 KV head across the TP axis, drop duplicate
    KV replicas, and fill the decode cache with the resulting two-head tensor.
    Decode operations delegate to the decode cache unchanged.
    """

    def __init__(self, tp4_cache: TTNNPagedAttentionKVCache, decode_cache: TTNNPagedAttentionKVCache, device):
        self.tp4_cache = tp4_cache
        self.decode_cache = decode_cache
        self.device = device
        self.num_layers = decode_cache.num_layers
        self.num_kv_heads = decode_cache.num_kv_heads
        self.head_dim = decode_cache.head_dim
        self.config = decode_cache.config

    def __getattr__(self, name):
        return getattr(self.decode_cache, name)

    def _tp4_kv_to_decode_kv(self, kv: ttnn.Tensor) -> ttnn.Tensor:
        target_heads = int(self.decode_cache.num_kv_heads)
        if _tp_degree(self.device) <= 1 or int(kv.shape[1]) == target_heads:
            return kv
        gathered = ttnn.all_gather(
            kv,
            dim=1,
            num_links=1,
            cluster_axis=_tp_cluster_axis(self.device),
            memory_config=ttnn.L1_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )
        if int(gathered.shape[1]) == target_heads:
            return gathered

        group = max(1, int(gathered.shape[1]) // target_heads)
        selected = []
        for head_idx in range(target_heads):
            src_idx = min(head_idx * group, int(gathered.shape[1]) - 1)
            selected.append(
                ttnn.slice(
                    gathered,
                    [0, src_idx, 0, 0],
                    [int(gathered.shape[0]), src_idx + 1, int(gathered.shape[2]), int(gathered.shape[3])],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            )
        decode_kv = ttnn.concat(selected, dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        for part in selected:
            ttnn.deallocate(part)
        ttnn.deallocate(gathered)
        return decode_kv

    def paged_fill_on_device(
        self, key_states: ttnn.Tensor, value_states: ttnn.Tensor, layer_idx: int, batch_idx: int = 0
    ):
        self.tp4_cache.paged_fill_on_device(key_states, value_states, layer_idx=layer_idx, batch_idx=batch_idx)
        decode_k = self._tp4_kv_to_decode_kv(key_states)
        decode_v = self._tp4_kv_to_decode_kv(value_states)
        self.decode_cache.paged_fill_on_device(decode_k, decode_v, layer_idx=layer_idx, batch_idx=batch_idx)
        if decode_k is not key_states:
            ttnn.deallocate(decode_k)
        if decode_v is not value_states:
            ttnn.deallocate(decode_v)

    def paged_update_on_device(self, *args, **kwargs):
        return self.decode_cache.paged_update_on_device(*args, **kwargs)

    def paged_sdpa_decode(self, *args, **kwargs):
        return self.decode_cache.paged_sdpa_decode(*args, **kwargs)

    def update_seq_length(self, layer_idx: int, seq_len: int = 1) -> None:
        self.tp4_cache.update_seq_length(layer_idx=layer_idx, seq_len=seq_len)
        self.decode_cache.update_seq_length(layer_idx=layer_idx, seq_len=seq_len)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.decode_cache.get_seq_length(layer_idx=layer_idx)

    def reset(self) -> None:
        self.tp4_cache.reset()
        self.decode_cache.reset()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TTNNDotsOCRPipeline(TTNNModule):
    """Standalone TTNN inference pipeline for dots.ocr.

    Orchestrates embedding, vision tower, decoder stack, final norm, and
    lm_head entirely through TTNN -- no HF ``model.generate()`` dependency.
    Scatter-merge and argmax run on device.

    Inherits from TTNNModule so that ``set_device()`` can recursively
    initialize all child modules (device, device_state, bypass flags,
    weight preprocessing).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        embedding: TTNNEmbedding,
        vision_tower: TTNNDotsOCRVisionTower,
        decoder_stack: TTNNDotsOCRLayerStack,
        final_norm: TTNNDistributedRMSNorm,
        lm_head: TTNNModule,
        paged_cache: TTNNPagedAttentionKVCache,
        graph_prefill: TTNNDotsOCRPrefillGraph,
        graph_decode: TTNNDotsOCRDecodeGraph,
        device: "ttnn.MeshDevice",
        config: PipelineConfig,
    ):
        super().__init__()
        self._bypass_tensor_wrapping = True

        self.embedding = embedding
        self.vision_tower = vision_tower
        self.decoder_stack = decoder_stack
        self.final_norm = final_norm
        self.prefill_decoder_stack = None
        self.decode_decoder_stack = None
        self.prefill_final_norm = None
        self.decode_final_norm = None
        self.lm_head = lm_head
        self.graph_prefill = graph_prefill
        self.graph_decode = graph_decode
        self.paged_cache = paged_cache
        self._device = device
        self.config = config
        self._batch_input_mapper = dp_batch_shard_tensor_mapper(device, config.batch_size)

        # Decode-loop buffer (allocated on first decode call, reused thereafter)
        self._decode_cache_position: Optional[ttnn.Tensor] = None
        self._decode_token_buffer: Optional[ttnn.Tensor] = None
        self._decode_token_buffer_has_next: bool = False
        self._decode_token_host: Optional[torch.Tensor] = None
        self._decode_cache_pos_host: Optional[torch.Tensor] = None
        self._decode_seq_counter: int = 0
        self._dp_readback_ring: Optional[List[ttnn.Tensor]] = None
        self._dp_readback_ring_n: int = 0

        # TP4-body paths can return post-body hidden from prefill and/or decode;
        # the final norm + lm_head + argmax then run eager via ``_eager_head``.
        self._external_prefill_head: bool = False
        self._external_decode_head: bool = False

    def _eager_head(self, hidden: ttnn.Tensor) -> ttnn.Tensor:
        """Final norm + lm_head + on-device argmax, run EAGER (outside any trace).

        Used by the TP4-body path: the full-vocab all_gather + argmax cannot share
        one captured trace with the heavy TP4 body (corrupts the argmax output
        buffer), so the graphs stop at the post-body hidden and this tail runs
        eager. ``hidden`` is the replicated full-H ``[B, 1, H]`` last-token state;
        the returned token tensor stays on device so decode feedback is unchanged.
        """
        h = self.final_norm.forward(hidden)
        logits = self.lm_head.forward(h)
        return _argmax_token_on_device(logits)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_hf_model(
        cls,
        model_path: str,
        device: "ttnn.MeshDevice",
        batch_size: int = 1,
    ) -> "TTNNDotsOCRPipeline":
        """Build pipeline from a HuggingFace model path.

        Loads the HF model, extracts components, creates TTNN modules,
        and assembles the pipeline.
        """
        from transformers import AutoModelForCausalLM

        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        # --- Create TTNN modules from HF components ---
        # _bypass_tensor_wrapping is set later by set_device() which sees
        # that the parent (the pipeline) is a TTNNModule and propagates
        # bypass=True to all children.

        embedding = TTNNEmbedding.from_torch(hf_model.model.embed_tokens)
        embedding._unique_name = "model.embed_tokens"

        vision_tower = TTNNDotsOCRVisionTower.from_torch(hf_model.vision_tower)

        vision_tower._unique_name = "vision_tower"
        vision_tower.override_children_module_names()

        text_body = _text_body_from_env(device)
        tp_decode_scheme = _decode_tp_scheme_from_env()

        def _build_symbiote_stack(norm_name: str = "model.norm"):
            decoder_layers = []
            for i, hf_layer in enumerate(hf_model.model.layers):
                layer = TTNNDotsOCRDecoderLayer.from_torch(hf_layer, tp_decode_scheme=tp_decode_scheme)
                layer._unique_name = f"model.layers.{i}"
                layer.override_children_module_names()
                # Mixed-precision decode gate_up: the first DOTS_OCR_GATE_UP_BF4_LAYERS
                # decoder layers keep the fast BFP4 fused gate_up weight; later layers use
                # BFP8 for higher precision. gate_up only -- down/o_proj are untouched. Must
                # run before move_weights so the weight loads at the chosen dtype.
                fgu = getattr(getattr(layer, "mlp", None), "fused_gate_up_proj", None)
                if fgu is not None and hasattr(fgu, "set_weight_dtype"):
                    gate_up_dtype = ttnn.bfloat4_b if i < DOTS_OCR_GATE_UP_BF4_LAYERS else ttnn.bfloat8_b
                    fgu.set_weight_dtype(gate_up_dtype)
                decoder_layers.append(layer)
            stack = TTNNDotsOCRLayerStack(decoder_layers)
            stack._unique_name = "model.layer_stack"
            norm_cls = TTNNDotsOCRLocalShardRMSNorm if tp_decode_scheme == "col_parallel" else TTNNDistributedRMSNorm
            norm = norm_cls.from_torch(hf_model.model.norm)
            norm._unique_name = norm_name
            return stack, norm

        dots_cfg = None
        prefill_stack = None
        prefill_norm = None
        decode_stack = None
        decode_norm = None
        use_tp4_prefill = text_body in {"tp4", "tp4_prefill"}
        use_tp4_decode = text_body == "tp4"

        if use_tp4_prefill:
            # Head-local replicated-hidden Megatron body from dots_ocr_tp4: fast
            # prefill (3 Q-heads / 1 KV-head per chip, no QKV gather). In
            # ``tp4_prefill`` mode only prefill uses this stack; decode stays on
            # the existing symbiote stack and consumes the adapter-filled cache.
            from models.experimental.dots_ocr_tp4.tt.common import DotsOCRConfig, tp4_lossy_matmul_dtype
            from models.experimental.dots_ocr_tp4.tt.model import DotsOCRPrefillModelTP4
            from models.experimental.dots_ocr_tp4.tt.rmsnorm import DotsOCRRMSNormTP4

            print(
                f"[dots_ocr] tp4 body lossy-matmul weight dtype (gate/up early + down + o_proj): "
                f"{tp4_lossy_matmul_dtype()} "
                f"(DOTS_OCR_TP4_HI_PRECISION={os.environ.get('DOTS_OCR_TP4_HI_PRECISION', '<unset>')!r})"
            )

            tp_deg = _tp_degree(device)
            if int(batch_size) > 1 and dp_batch_shard_tensor_mapper(device, batch_size) is None:
                raise ValueError(
                    f"DOTS_OCR_TEXT_BODY={text_body} batch_size={batch_size} requires a DP batch-sharded "
                    f"mesh (batch along mesh axis 0 or 1); mesh shape is {getattr(device, 'shape', None)}"
                )
            if tp_deg > 1 and (hf_model.config.num_attention_heads % tp_deg) != 0:
                raise ValueError(
                    f"DOTS_OCR_TEXT_BODY={text_body} needs num_attention_heads "
                    f"({hf_model.config.num_attention_heads}) divisible by TP degree ({tp_deg})"
                )
            num_devices = int(device.get_num_devices()) if hasattr(device, "get_num_devices") else 1
            print(
                f"[dots_ocr] text body: {text_body} "
                f"(tp4 prefill, num_devices={num_devices}, tp={tp_deg}, batch={batch_size})"
            )
            dots_cfg = DotsOCRConfig.from_hf(hf_model.config)
            prefill_stack = DotsOCRPrefillModelTP4.from_torch(
                device,
                dots_cfg,
                hf_model.model.layers,
                weight_dtype=ttnn.bfloat16,
            )
            prefill_stack._unique_name = "model.layer_stack_tp4_prefill"
            prefill_norm = DotsOCRRMSNormTP4.from_torch(device, hf_model.model.norm, eps=dots_cfg.rms_norm_eps)
            prefill_norm._unique_name = "model.norm_tp4_prefill"

        if use_tp4_decode:
            decode_stack = prefill_stack
            decode_norm = prefill_norm
            decoder_stack = prefill_stack
            final_norm = prefill_norm
        else:
            print(f"[dots_ocr] decode body: symbiote; decoder TP decode scheme: {tp_decode_scheme}")
            decode_stack, decode_norm = _build_symbiote_stack()
            decoder_stack = prefill_stack if use_tp4_prefill else decode_stack
            final_norm = prefill_norm if use_tp4_prefill else decode_norm
            if not use_tp4_prefill:
                prefill_stack = decode_stack
                prefill_norm = decode_norm

        # ``TTNNDotsOCRDRAMShardedLMHead`` lays the lm_head weight out
        # WIDTH_SHARDED across all 12 DRAM banks per chip and runs the matmul
        # with ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig``. The
        # decode lm_head is bandwidth-bound (~60-65% DRAM in Tracy), so the
        # 12-bank parallel read pulls per-chunk device time from
        # ~78-100 µs (auto-config, 1-bank serial) toward ~40-50 µs.
        #
        # The previous correctness regression (garbled output when this class
        # was wired in) was a weight-mapper bug: ``shard_tensor_to_mesh_mapper(
        # dim=-1)`` always sharded along the first mesh axis, so on T3K
        # mesh=(8,1) (DP=8, TP=1) every chip got 1/8 of the vocab columns,
        # ``_tp_requires_ccl`` was False so no final all_gather ran, and
        # argmax picked from a 1/8 garbage slice. The class now uses
        # ``_tp_mesh_mapper`` which replicates across the DP axis and shards
        # only across the TP axis -- so DP-only meshes get a fully-replicated
        # vocab on every chip and TP meshes still get the N-parallel split
        # plus all_gather restore.
        lm_head = TTNNDotsOCRDRAMShardedLMHead.from_torch(hf_model.lm_head)
        lm_head._unique_name = "lm_head"

        # Paged KV cache. The TP4 body stores exactly its one assigned KV head per
        # chip (head-sharded across the mesh). Hybrid prefill fills both that
        # cache and the symbiote two-KV-head decode cache.
        if use_tp4_decode:
            from models.experimental.dots_ocr_tp4.tt.kv_cache import create_paged_kv_cache as _create_tp4_paged_kv_cache

            paged_cache = _create_tp4_paged_kv_cache(dots_cfg, device, batch_size=batch_size)
        elif use_tp4_prefill:
            from models.experimental.dots_ocr_tp4.tt.kv_cache import create_paged_kv_cache as _create_tp4_paged_kv_cache

            tp4_paged_cache = _create_tp4_paged_kv_cache(dots_cfg, device, batch_size=batch_size)
            decode_paged_cache = _create_paged_kv_cache(hf_model.config, device, batch_size)
            paged_cache = _TP4PrefillDecodeCacheAdapter(tp4_paged_cache, decode_paged_cache, device)
        else:
            paged_cache = _create_paged_kv_cache(hf_model.config, device, batch_size)

        # Pipeline config (before graphs)
        eos_token_ids = [151643, 151673]
        if hasattr(hf_model, "generation_config") and hf_model.generation_config is not None:
            gc_eos = getattr(hf_model.generation_config, "eos_token_id", None)
            if gc_eos is not None:
                eos_token_ids = gc_eos if isinstance(gc_eos, list) else [gc_eos]

        config = PipelineConfig(
            vocab_size=hf_model.config.vocab_size,
            hidden_size=hf_model.config.hidden_size,
            image_token_id=getattr(hf_model.config, "image_token_id", 151665),
            eos_token_ids=eos_token_ids,
            num_devices=device.get_num_devices(),
            batch_size=batch_size,
        )

        _bim = dp_batch_shard_tensor_mapper(device, batch_size)
        # The TP4 prefill body consumes a replicated full-H stream; the embedding
        # / vision scatter still emit the H/num_devices column shard, so prefill
        # all-gathers to full-H at the body boundary.
        prefill_gather_hidden_for_full_body = use_tp4_prefill
        decode_gather_hidden_for_full_body = use_tp4_decode
        graph_prefill = TTNNDotsOCRPrefillGraph(
            prefill_stack,
            prefill_norm,
            lm_head,
            embedding=embedding,
            vision_tower=vision_tower,
            image_token_id=config.image_token_id,
            hidden_size=config.hidden_size,
            scatter_uses_dp_batch_mapper=_bim is not None,
            gather_hidden_for_full_body=prefill_gather_hidden_for_full_body,
        )
        graph_prefill._unique_name = "dots_ocr_graph_prefill"
        # Prefill text-only: token ids in ``graph_prefill`` embed inside one trace.
        # Multimodal: patch_embed runs once outside; vision trunk + scatter fuse +
        # decoder + argmax share one ``TracedRun`` (``forward`` takes four extra
        # device tensors: patch tokens, grid, gather index, image mask).
        #
        # Decode: embedding inside ``graph_decode``; ``@trace_enabled`` on
        # ``TTNNEmbedding`` falls through to ``forward()`` while the decode graph
        # trace is capturing (``_TRACE_RUNNING`` in ``run_config.TracedRun``).
        # Per-chip decode token input is ``[1,1]`` uint32 ROW_MAJOR; embedding
        # in the graph avoids ``_dp_repack_batch_sharded_hidden`` on decode_input.
        graph_decode = TTNNDotsOCRDecodeGraph(
            decode_stack,
            decode_norm,
            lm_head,
            embedding=embedding,
            gather_hidden_for_full_body=decode_gather_hidden_for_full_body,
            hidden_size=config.hidden_size,
        )
        graph_decode._unique_name = "dots_ocr_graph_decode"

        pipeline = cls(
            embedding=embedding,
            vision_tower=vision_tower,
            decoder_stack=decoder_stack,
            final_norm=final_norm,
            lm_head=lm_head,
            paged_cache=paged_cache,
            graph_prefill=graph_prefill,
            graph_decode=graph_decode,
            device=device,
            config=config,
        )
        pipeline._unique_name = "dots_ocr_pipeline"
        pipeline.prefill_decoder_stack = prefill_stack
        pipeline.decode_decoder_stack = decode_stack
        pipeline.prefill_final_norm = prefill_norm
        pipeline.decode_final_norm = decode_norm
        # TP4 graph paths return the post-body hidden; run the head eager only
        # for the graph(s) that use the TP4 body.
        pipeline._external_prefill_head = use_tp4_prefill
        pipeline._external_decode_head = use_tp4_decode

        # Set device and preprocess weights
        pipeline._set_device_and_preprocess(device)
        pipeline.prefill = timed_call(pipeline.prefill, "prefill", "TTNNDotsOCRPipelinePrefill")
        pipeline.decode_step = timed_call(pipeline.decode_step, "decode", "TTNNDotsOCRPipelineDecode")
        return pipeline

    # ------------------------------------------------------------------
    # Device / weight setup
    # ------------------------------------------------------------------

    def _set_device_and_preprocess(self, device: "ttnn.MeshDevice") -> None:
        """Recursively set device/device_state on all children, then preprocess weights."""
        from models.experimental.tt_symbiote.utils.device_management import set_device

        set_device(self, device, register_forward_hook=False, dump_visualization=False)

        # Preprocess and move weights for every leaf TTNNModule.
        for module in self._collect_ttnn_modules():
            module.preprocess_weights()
            module.move_weights_to_device()

        # LM head: HiFi2 + packer_l1_acc + FP32 dest accum (matches stable argmax path).
        self.lm_head.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _collect_ttnn_modules(self) -> List[TTNNModule]:
        """Recursively collect all TTNNModule instances from pipeline components."""
        found: List[TTNNModule] = []
        visited: set = set()

        def _recurse(obj):
            obj_id = id(obj)
            if obj_id in visited:
                return
            visited.add(obj_id)
            if isinstance(obj, TTNNModule):
                found.append(obj)
            for attr_name in dir(obj):
                if attr_name.startswith("_"):
                    continue
                try:
                    value = getattr(obj, attr_name)
                except Exception:
                    continue
                if isinstance(value, TTNNModule):
                    _recurse(value)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        if isinstance(item, TTNNModule):
                            _recurse(item)
                elif isinstance(value, dict):
                    for item in value.values():
                        if isinstance(item, TTNNModule):
                            _recurse(item)

        for component in [
            self.embedding,
            self.vision_tower,
            self.decoder_stack,
            self.prefill_decoder_stack,
            self.decode_decoder_stack,
            self.final_norm,
            self.prefill_final_norm,
            self.decode_final_norm,
            self.lm_head,
            self.graph_prefill,
            self.graph_decode,
        ]:
            if component is not None:
                _recurse(component)
        return found

    def _mesh_dp_dual_stream(self) -> bool:
        """True when batch is sharded one row per device (DP batch parallel)."""
        return self._batch_input_mapper is not None

    def _dp_repack_batch_sharded_hidden(self, tt_hidden: ttnn.Tensor) -> ttnn.Tensor:
        return _dp_repack_batch_sharded_hidden_for_device(self.device, self._batch_input_mapper, tt_hidden)

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def prefill(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> Union[int, List[int]]:
        """Run prefill (first forward pass) and return the first generated token(s).

        Args:
            input_ids: ``[1, S]`` or ``[B, S]`` int64/int32 token IDs on host.
                With DP mesh batch sharding (``B == num_devices``), use one row
                per independent stream; each device runs local batch 1.
            pixel_values: Optional vision input for multimodal prefill.
            image_grid_thw: Optional grid info for vision input.

        Returns:
            First predicted token ID (int), or a list of ``B`` IDs when
            ``_mesh_dp_dual_stream()`` is active.
        """
        if _tp4_prefill_body_enabled(self.device):
            _dots_ocr_signpost("dots_ocr.prefill_start")
        dual = self._mesh_dp_dual_stream()
        if dual and int(input_ids.shape[0]) != int(self.config.batch_size):
            raise ValueError(
                f"DP batch-parallel prefill expects input_ids batch {self.config.batch_size}, "
                f"got {input_ids.shape[0]}"
            )
        seq_len = input_ids.shape[-1]

        # --- Embedding ---
        # All children have _bypass_tensor_wrapping=True (pipeline is a
        # TTNNModule parent), so we convert input_ids to ttnn ourselves.
        id_mapper = (
            self._batch_input_mapper
            if self._batch_input_mapper is not None
            else ttnn.ReplicateTensorToMesh(self.device)
        )
        with _profile_stage(self.device, "prefill.input_ids_h2d"):
            # Upload directly as uint32 (token ids are non-negative, so the
            # int32->uint32 reinterpret is lossless). ttnn.embedding requires
            # uint32 indices; handing them in as uint32 here lets
            # TTNNEmbedding.forward skip its pad -> typecast -> slice block
            # (the int32 path it would otherwise take for non-tile seq_len).
            tt_input_ids = ttnn.from_torch(
                input_ids.to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                mesh_mapper=id_mapper,
            )

        # --- Multimodal vision: patch_embed outside; vision trunk + scatter + decoder in one trace ---
        # Set when the dual multimodal path scatter-fuses + gathers EAGER and hands
        # the captured graph a plain full-H hidden (no in-trace scatter/CCL).
        prefused_multimodal = False
        if pixel_values is not None:
            if image_grid_thw is None:
                raise ValueError("image_grid_thw is required when pixel_values is set")
            num_devices = int(self.device.get_num_devices()) if hasattr(self.device, "get_num_devices") else 1
            with _profile_stage(self.device, "prefill.vision_patch_embed"):
                x_patch = self.vision_tower.patch_embed(pixel_values, image_grid_thw)
                if isinstance(x_patch, torch.Tensor):
                    mapper = ttnn.ReplicateTensorToMesh(self.device) if num_devices > 1 else None
                    x_patch = x_patch.unsqueeze(1) if x_patch.dim() == 3 else x_patch
                    x_patch = ttnn.from_torch(
                        x_patch.to(torch.bfloat16),
                        device=self.device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=mapper,
                    )
                if len(x_patch.shape) == 3:
                    x_patch = ttnn.reshape(x_patch, (1, 1, int(x_patch.shape[1]), int(x_patch.shape[2])))
                actual_vision_seq_len = int(x_patch.shape[2])
                vision_bucket = (
                    self.vision_tower.block_stack.nearest_bucket(actual_vision_seq_len)
                    if self.vision_tower.block_stack is not None and self.vision_tower._trace_enabled
                    else -1
                )
                use_vision_sdpa_mask = os.environ.get("DOTS_OCR_USE_FULL_SDPA_MASK", "").lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                vision_attention_mask = (
                    self.vision_tower.build_padded_attention_mask(actual_vision_seq_len, vision_bucket)
                    if use_vision_sdpa_mask and vision_bucket != -1
                    else None
                )
            n_vis = self.vision_tower.merged_vision_sequence_length(image_grid_thw, pixel_values)
            tt_idx, tt_mask = self.graph_prefill.get_or_build_scatter_tensors(
                input_ids, n_vis, id_mapper, _tp_degree(self.device)
            )
            grid_cpu = image_grid_thw.detach().cpu()
            if grid_cpu.dim() == 1:
                grid_cpu = grid_cpu.unsqueeze(0)
            mm_grid_thw = _normalize_image_grid_thw_torch(grid_cpu)
            # Vision trunk runs EAGER here, OUTSIDE the captured prefill trace. It
            # does host-side reads (dynamic cu_seqlens / grid) that are illegal
            # during trace capture and its large attention CBs clash with the
            # trace buffer's pinned L1. Only its fixed-shape output (vision_tt) is
            # handed to the trace, which then records just scatter-fuse + decoder
            # body. Vision is image-dependent and ~90% of prefill, so tracing it
            # would buy little and force a re-bucket per image anyway.
            #
            # ``disable_trace`` pins ``_TRACE_RUNNING`` so the @trace_enabled
            # ``vision_tower.block_stack`` runs eager instead of capturing its OWN
            # trace -- preserving the old in-graph behavior (vision was always run
            # under the prefill graph's _TRACE_RUNNING=True). A separate vision
            # trace would stay "active" and make the prefill capture's buffer
            # allocations illegal ("Writes are not supported during trace capture").
            with _profile_stage(self.device, "prefill.vision_trunk"):
                vision_tt = disable_trace(self.vision_tower.forward_post_patch_embed)(
                    x_patch, mm_grid_thw, attention_mask=vision_attention_mask
                )
            # Pre-build the scatter zero-row EAGERLY (outside the trace). It is the
            # only host upload (ttnn.from_torch) on the captured prefill path; if it
            # were built lazily inside _scatter_fuse_text_and_vision during capture
            # it triggers "Writes are not supported during trace capture" (one per
            # mesh device). Building it here (idempotent, cached on the graph) keeps
            # the captured forward host-transfer-free.
            self.graph_prefill._ensure_scatter_zero_row(
                _tp_degree(self.device), self.graph_prefill._hidden_size, int(vision_tt.shape[3])
            )
            if dual:
                with _profile_stage(self.device, "prefill.text_embedding"):
                    # Run the @trace_enabled embedding EAGER (disable_trace) so it
                    # does not capture its own sub-trace. A resident embedding trace
                    # is "active" while the prefill graph's input buffers are set up
                    # ("Allocating device buffers is unsafe due to the existence of
                    # an active trace"), which makes the prefill capture's transfers
                    # illegal. Folding it into the eager pre-graph region leaves the
                    # prefill graph as the sole captured trace, fed a plain tensor.
                    text_embeds = disable_trace(self.embedding)(tt_input_ids)
                if self._batch_input_mapper is not None and int(text_embeds.shape[0]) > 1:
                    with _profile_stage(self.device, "prefill.dp_repack_hidden"):
                        text_embeds = self._dp_repack_batch_sharded_hidden(text_embeds)
                # Scatter-fuse vision into the text embeds and gather to full width
                # EAGER, OUTSIDE the captured trace. The scatter's gather/all_gather
                # are plain CCL ops that do per-device host writes (semaphore setup)
                # which are illegal during trace capture (bisected: scatter alone
                # captures clean, +all_gather_hidden FATALs). Producing the full-H
                # fused hidden here lets the captured graph record ONLY the decoder
                # body, fed a plain replicated tensor (no mm_args / scatter / CCL).
                with _profile_stage(self.device, "prefill.scatter_fuse"):
                    fused = self.graph_prefill._scatter_fuse_text_and_vision(text_embeds, vision_tt, tt_idx, tt_mask)
                    if self.graph_prefill._gather_hidden_for_full_body:
                        fused = _all_gather_hidden_to_full(fused, self.device, self.graph_prefill._hidden_size)
                hidden_states = fused
                prefused_multimodal = True
            else:
                hidden_states = tt_input_ids
        elif dual:
            with _profile_stage(self.device, "prefill.text_embedding"):
                text_embeds = self.embedding(tt_input_ids)
            hidden_states = text_embeds
        else:
            hidden_states = tt_input_ids
        if (
            dual
            and self._batch_input_mapper is not None
            and pixel_values is None
            and isinstance(hidden_states, ttnn.Tensor)
            and len(hidden_states.shape) == 3
            and int(hidden_states.shape[0]) > 1
        ):
            with _profile_stage(self.device, "prefill.dp_repack_hidden"):
                hidden_states = self._dp_repack_batch_sharded_hidden(hidden_states)

        # --- Cache position for prefill ---
        with _profile_stage(self.device, "prefill.cache_position_h2d"):
            cache_position = torch.arange(0, seq_len, dtype=torch.int32)
            tt_cache_position = ttnn.from_torch(
                cache_position,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Traced prefill graph. For the prefused multimodal path (and text-only) the
        # graph records ONLY the decoder body, fed a plain replicated hidden -- vision,
        # embed, scatter-fuse and the full-H all_gather all ran EAGER above (they do
        # host reads / plain-CCL host writes that are illegal during trace capture).
        # The non-prefused multimodal path still passes vision tensors in (in-graph
        # scatter); that path is not trace-captured.
        with _profile_stage(self.device, "prefill.graph_prefill_sync"):
            if pixel_values is not None and not prefused_multimodal:
                token_id_tt = self.graph_prefill(
                    hidden_states,
                    tt_cache_position,
                    vision_tt,
                    tt_idx,
                    tt_mask,
                    past_key_value=self.paged_cache,
                )
            else:
                token_id_tt = self.graph_prefill(hidden_states, tt_cache_position, past_key_value=self.paged_cache)

        # TP4 prefill body: the graph returned the post-body hidden; run the head eager.
        if self._external_prefill_head:
            token_id_tt = self._eager_head(token_id_tt)

        # --- Read to host ---
        with _profile_stage(self.device, "prefill.first_token_readback"):
            if dual:
                token_composer = _token_mesh_composer(self.device, self.config.batch_size, dual)
                token_id_torch = ttnn.to_torch(
                    token_id_tt,
                    mesh_composer=token_composer,
                )
                token_id_torch = _select_dp_token_replicas(token_id_torch, self.device, self.config.batch_size)
            else:
                token_id_torch = _first_device_token_to_torch(token_id_tt)
        flat_ids = [int(x) for x in token_id_torch.reshape(-1).tolist()]
        if dual:
            if len(flat_ids) != int(self.config.batch_size):
                raise RuntimeError(
                    f"Expected {self.config.batch_size} first tokens from mesh concat, got {len(flat_ids)}"
                )
            first_out: Union[int, List[int]] = flat_ids
        else:
            first_out = flat_ids[0]

        # Clean up prefill-only tensors
        ttnn.deallocate(tt_cache_position)

        _dots_ocr_signpost("dots_ocr.prefill_end")
        return first_out

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def _init_decode_buffers(self, prev_token_id: Union[int, List[int]]):
        """Allocate reusable device buffers for decode loop on first call."""
        if isinstance(prev_token_id, int):
            self._decode_token_host = torch.tensor([[prev_token_id]], dtype=torch.int32)
        else:
            self._decode_token_host = torch.tensor(prev_token_id, dtype=torch.int32).reshape(len(prev_token_id), 1)
        token_mapper = (
            self._batch_input_mapper
            if self._batch_input_mapper is not None
            else ttnn.ReplicateTensorToMesh(self.device)
        )
        self._decode_token_buffer = ttnn.from_torch(
            self._decode_token_host,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=token_mapper,
        )
        self._decode_token_buffer_has_next = True
        self._decode_seq_counter = self.paged_cache.get_seq_length(layer_idx=0)
        self._decode_cache_pos_host = torch.tensor([self._decode_seq_counter], dtype=torch.int32)
        # Scalar cache position: always replicate (same global index on every
        # device). Do not use ``_batch_input_mapper`` here: ND shard expects one
        # host chunk per mesh device, which a length-1 tensor does not satisfy.
        self._decode_cache_position = ttnn.from_torch(
            self._decode_cache_pos_host,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def decode_step(
        self,
        prev_token_id: Union[int, List[int]],
        read_from_device: bool = True,
        readback_dram_slot: Optional[ttnn.Tensor] = None,
    ) -> Union[int, List[int], ttnn.Tensor, None]:
        """Execute one decode step entirely on device.

        Args:
            prev_token_id: Previous token ID(s) on host: ``int`` for batch 1, or
                a list of length ``B`` for DP batch-parallel streams.

        Returns:
            Next predicted token ID, or a list of ``B`` IDs when dual-stream
            DP batch mode is active.
        """
        if self._decode_token_buffer is None:
            with _profile_stage(self.device, "decode.init_buffers"):
                self._init_decode_buffers(prev_token_id)
        else:
            # ``has_next`` is True whenever the previous iter's argmax already
            # wrote the next-token directly into ``_decode_token_buffer`` via
            # ``ttnn.copy(token_id_tt, _decode_token_buffer)`` below. In that
            # case the host upload is redundant -- the device already has the
            # token. This holds for **both** non-DP single-stream and DP
            # batch-sharded modes: per chip, ``token_id_tt`` and
            # ``_decode_token_buffer`` are both [1,1] uint32 ROW_MAJOR, so
            # the on-device copy is just a per-chip element copy and the
            # mesh-sharding metadata stays consistent.
            #
            # NOTE: this flag is the **only** DP host-overhead reduction
            # being landed in this pass (step 1 of the bisect after the
            # malloc-corrupted multi-change attempt). Embedding-in-trace
            # and pipelined readback stay disabled for DP until step 1 is
            # confirmed clean across two ``generate()`` calls.
            upload_prev_token = not self._decode_token_buffer_has_next
            if upload_prev_token:
                if isinstance(prev_token_id, int):
                    self._decode_token_host[0][0] = prev_token_id
                else:
                    for row, tid in enumerate(prev_token_id):
                        self._decode_token_host[row][0] = tid
            if upload_prev_token and self._batch_input_mapper is not None:
                # Mesh-sharded device buffer does not match host logical shape [B,1];
                # upload with the same mapper, then device-to-device copy into the
                # preallocated decode buffer (for trace-stable tensor identity).
                with _profile_stage(self.device, "decode.prev_token_h2d_dp"):
                    token_upload = ttnn.from_torch(
                        self._decode_token_host,
                        dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        device=self.device,
                        mesh_mapper=self._batch_input_mapper,
                    )
                    ttnn.copy(token_upload, self._decode_token_buffer)
                    ttnn.deallocate(token_upload)
            elif upload_prev_token:
                with _profile_stage(self.device, "decode.prev_token_h2d"):
                    token_host_tt = ttnn.from_torch(
                        self._decode_token_host,
                        dtype=ttnn.uint32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                    ttnn.copy_host_to_device_tensor(token_host_tt, self._decode_token_buffer)

            self._decode_seq_counter += 1
            # DEBUG: log generation step + the cache_position SDPA will see
            # on the NEXT decode iteration. Throttled to keep host output
            # readable. ``_decode_seq_counter`` is the next-token cache pos.
            _gen_step = self._decode_seq_counter - int(self._decode_cache_pos_host.item())
            if _gen_step == 1 or _gen_step % 100 == 0:
                print(
                    f"[decode] gen_step={_gen_step}  " f"cache_position={self._decode_seq_counter}",
                    flush=True,
                )
            # Device-side cache_position increment for both DP and non-DP.
            # ``_decode_cache_position`` is REPLICATED across the mesh (a
            # single global counter, see ``_init_decode_buffers``) so the
            # ``ttnn.add`` runs identically on every chip and stays in sync
            # without any host-side h2d. Used to be DP-gated to a host h2d
            # path (``_batch_input_mapper is not None``); removing the gate
            # cuts another per-iter ``ttnn.from_torch + copy_host_to_device``
            # (~3-5 ms wall-clock at DP=8 on T3K).
            with _profile_stage(self.device, "decode.cache_position_device_inc"):
                # NOTE: tried ``ttnn.add(..., output_tensor=cache_position)``
                # to eliminate the explicit ``ttnn.copy`` + ``ttnn.deallocate``,
                # but the in-place output_tensor path raises
                # ``Optional output tensor with Row Major input is not
                # supported right now for Elementwise operations``
                # (ttnn/cpp/ttnn/operations/eltwise/binary/binary.cpp:362)
                # because ``_decode_cache_position`` is INT32 / ROW_MAJOR. Stay
                # on the temp-buffer path until in-place row-major elementwise
                # is supported.
                cache_next = ttnn.add(self._decode_cache_position, 1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.copy(cache_next, self._decode_cache_position)
                ttnn.deallocate(cache_next)

        trace_decode_tokens = getattr(self.graph_decode, "_d_embedding", None) is not None
        if trace_decode_tokens:
            decode_input = self._decode_token_buffer
        else:
            with _profile_stage(self.device, "decode.embedding"):
                decode_input = self.embedding(self._decode_token_buffer)
            if self._mesh_dp_dual_stream() and self._batch_input_mapper is not None:
                with _profile_stage(self.device, "decode.dp_repack_hidden"):
                    decode_input = self._dp_repack_batch_sharded_hidden(decode_input)

        with _profile_stage(self.device, "decode.graph_decode_sync"):
            token_id_tt = self.graph_decode(decode_input, self._decode_cache_position, past_key_value=self.paged_cache)
            # TP4 decode body: graph returned the post-body hidden; run the head eager
            # (norm + lm_head + argmax). Token stays on device, so the on-device
            # feedback copy below is unchanged.
            if self._external_decode_head:
                token_id_tt = self._eager_head(token_id_tt)
        # On-device token feedback. Skip the per-iter ``ttnn.from_torch +
        # ttnn.copy_host_to_device_tensor`` host upload by writing the
        # next-iter input directly on device.
        #
        # Per chip, both ``token_id_tt`` (argmax output reshaped to ``[b,1]``
        # in ``_argmax_token_on_device``) and ``_decode_token_buffer`` are
        # ``uint32`` ``ROW_MAJOR`` with the same physical ``[1,1]`` shape per
        # chip (DP shards along batch dim, replicated metadata otherwise),
        # so the copy is a per-chip element copy and mesh-sharding metadata
        # is preserved.
        if readback_dram_slot is not None:
            ttnn.copy(token_id_tt, readback_dram_slot)
        ttnn.copy(token_id_tt, self._decode_token_buffer)
        self._decode_token_buffer_has_next = True
        token_id_snapshot = None
        if self._batch_input_mapper is None and not read_from_device:
            # Pipelined async d2h via ``.cpu(blocking=False)`` is enabled
            # **only** for non-DP single-stream / replicated-mesh decode.
            # Tried for DP (step 3 of the bisect) but reproducibly crashed
            # with ``malloc(): unaligned tcache chunk detected`` on the
            # warmup pass 2 prefill ``from_torch(input_ids)`` -- the
            # multi-device mesh-sharded snapshot lifecycle is broken in
            # this path. Stay gated until a safer DP-aware mechanism
            # (ttnn events / pre-allocated host buffer pool) is wired up.
            token_id_snapshot = token_id_tt.cpu(blocking=False)

        if readback_dram_slot is not None:
            if read_from_device:
                raise RuntimeError("readback_dram_slot is only for deferred host readback (read_from_device=False)")
            return None

        if not read_from_device:
            if token_id_snapshot is None:
                raise RuntimeError("Deferred decode readback did not produce a token snapshot")
            return token_id_snapshot

        # --- Read to host ---
        with _profile_stage(self.device, "decode.token_readback"):
            if self._mesh_dp_dual_stream():
                token_composer = _token_mesh_composer(self.device, self.config.batch_size, True)
                token_id_torch = ttnn.to_torch(
                    token_id_tt,
                    mesh_composer=token_composer,
                )
                token_id_torch = _select_dp_token_replicas(token_id_torch, self.device, self.config.batch_size)
            else:
                token_id_torch = _first_device_token_to_torch(token_id_tt)
        flat_ids = [int(x) for x in token_id_torch.reshape(-1).tolist()]
        if self._mesh_dp_dual_stream():
            return flat_ids
        return flat_ids[0]

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        stop_on_eos: bool = True,
    ) -> Union[List[int], List[List[int]]]:
        """Full generation: prefill + decode loop.

        Args:
            input_ids: ``[1, S]`` or ``[B, S]`` token IDs on host (``B`` streams
                for DP mesh batch sharding).
            pixel_values: Optional vision input.
            image_grid_thw: Optional grid info for vision.
            max_new_tokens: Maximum number of tokens to generate.
            stop_on_eos: Stop appending tokens once EOS is produced. Disable
                this for fixed-depth performance runs.

        Returns:
            List of generated token IDs per stream (prompt excluded). Single
            stream: ``List[int]``. DP batch-parallel: ``List[List[int]]`` with
            one inner list per stream (same decode depth; each stream stops
            appending on EOS unless ``stop_on_eos`` is disabled).
        """
        _dots_ocr_signpost("dots_ocr.model_start")
        return self._generate_impl(
            input_ids,
            pixel_values,
            image_grid_thw,
            max_new_tokens,
            stop_on_eos,
        )

    def _generate_impl(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        max_new_tokens: int,
        stop_on_eos: bool,
    ) -> Union[List[int], List[List[int]]]:
        # Reset cache for fresh generation
        self.paged_cache.reset()
        self._decode_cache_position = None
        self._decode_token_buffer = None
        self._decode_token_buffer_has_next = False
        self._decode_seq_counter = 0
        self._dp_readback_ring = None
        first_out = self.prefill(input_ids, pixel_values, image_grid_thw)

        if max_new_tokens <= 1:
            if self._mesh_dp_dual_stream():
                if not isinstance(first_out, list):
                    raise RuntimeError("prefill must return a list of first tokens in DP dual-stream mode")
                return [[t] for t in first_out]
            if not isinstance(first_out, int):
                raise RuntimeError("prefill must return a single int in non-DP mode")
            return [first_out]

        if _tp4_prefill_body_enabled(self.device):
            _dots_ocr_signpost("dots_ocr.decode_start")
        try:
            return self._decode_after_prefill(first_out, max_new_tokens, stop_on_eos)
        finally:
            _dots_ocr_signpost("dots_ocr.decode_end")

    def _decode_after_prefill(
        self,
        first_out: Union[int, List[int]],
        max_new_tokens: int,
        stop_on_eos: bool,
    ) -> Union[List[int], List[List[int]]]:
        """Autoregressive decode loop (``max_new_tokens > 1``; prefill already ran)."""
        if self._mesh_dp_dual_stream():
            if not isinstance(first_out, list):
                raise RuntimeError("prefill must return a list of first tokens in DP dual-stream mode")
            currents: List[int] = list(first_out)
            generated: List[List[int]] = [[t] for t in currents]
            active = [True] * len(currents)
            num_streams = len(currents)
            rb_k = _dp_readback_batch_size()
            if rb_k <= 1:
                for _ in range(max_new_tokens - 1):
                    if stop_on_eos and not any(active):
                        break
                    next_toks = self.decode_step(currents)
                    if not isinstance(next_toks, list):
                        raise RuntimeError("decode_step must return a list in DP dual-stream mode")
                    for i in range(num_streams):
                        if stop_on_eos and not active[i]:
                            continue
                        generated[i].append(next_toks[i])
                        if stop_on_eos and next_toks[i] in self.config.eos_token_ids:
                            active[i] = False
                    currents = list(next_toks)
                return generated

            proto = ttnn.from_torch(
                torch.zeros(num_streams, 1, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                mesh_mapper=self._batch_input_mapper,
            )
            slots = [ttnn.clone(proto, memory_config=ttnn.DRAM_MEMORY_CONFIG) for _ in range(rb_k)]
            ttnn.deallocate(proto)
            self._dp_readback_ring = slots

            total_decode = max_new_tokens - 1
            decodes_done = 0
            for it in range(total_decode):
                if stop_on_eos and not any(active):
                    break
                if it > 0 and it % rb_k == 0:
                    ttnn.synchronize_device(self.device)
                    for sj in range(rb_k):
                        tok_torch = ttnn.to_torch(
                            slots[sj],
                            mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
                        )
                        flat = [int(x) for x in tok_torch.reshape(-1).tolist()]
                        if len(flat) != num_streams:
                            raise RuntimeError("batched DP readback token width mismatch")
                        for bi in range(num_streams):
                            if stop_on_eos and not active[bi]:
                                continue
                            tid = flat[bi]
                            generated[bi].append(tid)
                            if stop_on_eos and tid in self.config.eos_token_ids:
                                active[bi] = False
                        currents = list(flat)
                si = it % rb_k
                self.decode_step(
                    currents,
                    read_from_device=False,
                    readback_dram_slot=slots[si],
                )
                decodes_done += 1

            rem = decodes_done % rb_k
            if rem == 0 and decodes_done > 0:
                rem = rb_k
            if rem > 0:
                ttnn.synchronize_device(self.device)
                start = (decodes_done - rem) % rb_k
                for j in range(rem):
                    sj = (start + j) % rb_k
                    tok_torch = ttnn.to_torch(
                        slots[sj],
                        mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
                    )
                    flat = [int(x) for x in tok_torch.reshape(-1).tolist()]
                    if len(flat) != num_streams:
                        raise RuntimeError("batched DP readback token width mismatch")
                    for bi in range(num_streams):
                        if stop_on_eos and not active[bi]:
                            continue
                        tid = flat[bi]
                        generated[bi].append(tid)
                        if stop_on_eos and tid in self.config.eos_token_ids:
                            active[bi] = False
                    currents = list(flat)

            for s in slots:
                ttnn.deallocate(s)
            self._dp_readback_ring = None
            return generated

        if not isinstance(first_out, int):
            raise RuntimeError("prefill must return a single int in non-DP mode")
        first_token_id = first_out
        generated_single: List[int] = [first_token_id]
        current_token = first_token_id

        if _defer_decode_readback_enabled() and max_new_tokens > 1:
            if self._batch_input_mapper is not None:
                raise RuntimeError("DOTS_OCR_DEFER_DECODE_READBACK is only supported for non-DP single-stream decode")
            deferred_tokens: List[ttnn.Tensor] = []
            for _ in range(max_new_tokens - 1):
                token_tt = self.decode_step(current_token, read_from_device=False)
                if not isinstance(token_tt, ttnn.Tensor):
                    raise RuntimeError("Deferred decode step must return a TTNN token tensor")
                deferred_tokens.append(token_tt)
            with _profile_stage(self.device, "decode.deferred_token_readback"):
                ttnn.synchronize_device(self.device)
                for token_tt in deferred_tokens:
                    token_torch = ttnn.to_torch(
                        token_tt,
                        mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
                    )
                    generated_single.append(int(token_torch.reshape(-1)[0].item()))
                    ttnn.deallocate(token_tt)
            for idx, token in enumerate(generated_single):
                if token in self.config.eos_token_ids:
                    return generated_single[: idx + 1]
            return generated_single

        # 1-deep pipelined readback: issue iter N+1's trace replay before
        # resolving iter N's token. Removes the per-token sync wall caused
        # by ``ttnn.to_torch`` blocking until the d2h copy of the previous
        # token completes. Safe because:
        #   * ``decode_step`` already does an on-device
        #     ``ttnn.copy(token_id_tt, _decode_token_buffer)`` before
        #     issuing the d2h, so the next trace replay's input is set
        #     without needing a host roundtrip.
        #   * In steady-state non-DP single-stream decode the
        #     ``prev_token_id`` arg to ``decode_step`` is unused (gated on
        #     ``not _decode_token_buffer_has_next or _batch_input_mapper``).
        #   * tt-metal serializes a queued d2h copy of ``token_id_tt``
        #     before the next trace replay's write to that same buffer, so
        #     the host snapshot captures the correct token even though
        #     the device tensor is reused.
        # EOS still early-exits; worst case we executed exactly 1 extra
        # decode iteration past the EOS token before discarding it.
        if _pipeline_decode_readback_enabled() and max_new_tokens > 1 and self._batch_input_mapper is None:
            prev_snapshot: Optional[ttnn.Tensor] = None
            for _ in range(max_new_tokens - 1):
                cur_snapshot = self.decode_step(current_token, read_from_device=False)
                if not isinstance(cur_snapshot, ttnn.Tensor):
                    raise RuntimeError("decode_step must return a TTNN snapshot in pipelined readback")
                if prev_snapshot is not None:
                    prev_torch = ttnn.to_torch(
                        prev_snapshot,
                        mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
                    )
                    next_token = int(prev_torch.reshape(-1)[0].item())
                    ttnn.deallocate(prev_snapshot)
                    generated_single.append(next_token)
                    if next_token in self.config.eos_token_ids:
                        ttnn.deallocate(cur_snapshot)
                        return generated_single
                prev_snapshot = cur_snapshot
            if prev_snapshot is not None:
                prev_torch = ttnn.to_torch(
                    prev_snapshot,
                    mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0),
                )
                next_token = int(prev_torch.reshape(-1)[0].item())
                ttnn.deallocate(prev_snapshot)
                generated_single.append(next_token)
            return generated_single

        for _ in range(max_new_tokens - 1):
            next_tok = self.decode_step(current_token)
            if isinstance(next_tok, list):
                raise RuntimeError("decode_step returned a list in single-stream mode")
            next_token = next_tok
            generated_single.append(next_token)

            if next_token in self.config.eos_token_ids:
                break

            current_token = next_token

        return generated_single

    # ------------------------------------------------------------------
    # Argmax on device
    # ------------------------------------------------------------------

    def _argmax_on_device(self, logits: ttnn.Tensor) -> ttnn.Tensor:
        """Run argmax on device.

        Args:
            logits: ``[1, 1, vocab_size]`` bf16 TILE_LAYOUT on device (rank 3).

        Returns:
            token_id tensor on device.
        """
        return _argmax_token_on_device(logits)

    # ------------------------------------------------------------------
    # Warmup / trace management
    # ------------------------------------------------------------------

    def warmup(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> None:
        """Run warmup passes to prime JIT and capture traces.

        Call sequence:
          1. Warmup run (no trace): primes TTNN JIT, CCL, allocators.
          2. TracedRun.release_all(): release any traces from warmup.
          3. Trace capture run: captures decode trace.
          4. Reset: clear KV cache, reset positions.
        """
        # Pass 1: Warmup (TracedRun phase 1 -- no trace capture)
        self.generate(input_ids, pixel_values, image_grid_thw, max_new_tokens=2)
        self.paged_cache.reset()
        self._decode_cache_position = None

        # Release all traces so run 2 starts clean
        TracedRun.release_all()

        # Pass 2: Trace capture (TracedRun phase 2)
        self.generate(input_ids, pixel_values, image_grid_thw, max_new_tokens=4)
        self.paged_cache.reset()
        self._decode_cache_position = None

    def release(self) -> None:
        """Release all traced runs and deallocate pre-allocated buffers."""
        TracedRun.release_all()
        self._decode_cache_position = None
        self.graph_prefill.release_scatter_cache()
