# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``TtV4PrefillRuntime``: the engine-facing runtime of the DeepSeek-V4-Flash prefill (the structural contract of
``models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md`` section 2).

The engine owns the caches (``V4FlashKvCaches``), the chunk loop, the sockets and the migration comms; this runtime
owns the model and knows how to write the caches. Eager only (no trace) in this milestone.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
from models.demos.deepseek_v3_d_p.tt.v4.kv_table import build_v4_kv_chunk_table
from models.demos.deepseek_v3_d_p.tt.v4.transformer import TtV4PrefillTransformer


def _clear_export_caches(kv_caches, why: str) -> None:
    """DS4F-0271 (launch 20/21c audit): the compile warm-up and the capture warm chunk run a token-0 chunk through every layer
    AND its export writes, so the migration caches hold non-zero "token-0 text" rows up to max_seq before the first request;
    the driver's row range then exports whatever sits past a prompt's real rows (the HCA config is copied whole). Zero every
    export cache on device (the allocation-time kernel, no host transfer) once the warm-ups are done. Never inside a capture.
    """
    if kv_caches is None:
        return
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import DRAMZeroFill

    n = 0
    for name in ("swa_window", "hca_unified", "csa_unified", "csa_index_k", "csa_pending", "hca_pending"):
        t = getattr(kv_caches, name, None)
        if t is not None:
            DRAMZeroFill.op(t)
            n += 1
    logger.info(f"[v4 runtime] export caches zeroed ({n} tensors) after {why}")


@dataclass
class TtV4PrefillRuntimeConfig:
    chunk_size: int
    max_seq_len: int
    first_layer_idx: int
    num_layers: int
    is_first_rank: bool
    is_last_rank: bool
    num_users: int
    mesh_shape: tuple
    sp_axis: int = 0
    tp_axis: int = 1
    kv_only_last_layer: bool = True
    weight_cache_path: Optional[Path] = None
    # Trace islands (tt/v4/trace_island.py): the mHC sites, norms and MoE of every layer captured once and replayed per
    # chunk; the attention core stays eager. Needs the mesh opened with a trace region (PREFILL_USE_TRACE=1 ->
    # PREFILL_TRACE_REGION_SIZE, 256 MB default). The runner calls capture_trace() after the D2D endpoints exist.
    use_trace: bool = False

    @property
    def sp_factor(self) -> int:
        return int(self.mesh_shape[self.sp_axis])

    @property
    def tp_factor(self) -> int:
        return int(self.mesh_shape[self.tp_axis])


class TtV4PrefillRuntime:
    def __init__(
        self,
        mesh_device,
        hf_config,
        config: TtV4PrefillRuntimeConfig,
        *,
        layer_weights: Callable[[int], dict],
        top_level_weights: Optional[dict],
        num_links: int = 2,
        topology=ttnn.Topology.Linear,
        num_routed_experts: Optional[int] = None,
        dispatch_buffer_capacity_factor: int = 2,
    ):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.config = config
        assert config.max_seq_len % config.chunk_size == 0, (config.max_seq_len, config.chunk_size)
        # the hash-routed MoE layers (0..2) need the token ids, which only the first rank has
        assert (
            config.first_layer_idx == 0 or config.first_layer_idx >= 3
        ), "hash-MoE layers 0..2 must sit on the first rank"
        self.model = TtV4PrefillTransformer(
            mesh_device,
            hf_config,
            layer_weights=layer_weights,
            top_level_weights=top_level_weights,
            first_layer_idx=config.first_layer_idx,
            num_layers=config.num_layers,
            is_first_rank=config.is_first_rank,
            is_last_rank=config.is_last_rank,
            kv_only_last_layer=config.kv_only_last_layer,
            chunk_tokens=config.chunk_size,
            sp_axis=config.sp_axis,
            tp_axis=config.tp_axis,
            topology=topology,
            num_links=num_links,
            num_routed_experts=num_routed_experts,
            dispatch_buffer_capacity_factor=dispatch_buffer_capacity_factor,
            weight_cache_path=config.weight_cache_path,
        )
        self._pending_ids: dict = (
            {}
        )  # id(input tensor) -> (tensor, host token ids) between make_chunk_input and prefill_chunk
        self._needs_token_ids = config.is_first_rank and any(
            hf_config.mlp_layer_types[i] == "hash_moe"
            for i in range(config.first_layer_idx, config.first_layer_idx + config.num_layers)
        )
        self._ack: Optional[Callable[[int], None]] = None
        self._sink = None
        self._request_id = 0
        self._compiled = False
        self._ids_dev = None

    # ---- engine contract ----------------------------------------------------------------------------------------
    def compile(self, kv_caches) -> None:
        """Allocate every slot's chunk state, then run one warm chunk (slot 0, zeros) so the per-chunk loop hits no
        first-run compile; the slot state is reset afterwards."""
        c = self.config
        self.model.alloc_states(c.num_users, c.max_seq_len)
        self._log_dram("after alloc_states")
        x = self.make_chunk_input([0] * c.chunk_size)
        # every chunk index: the CSA attention's live-extent programs differ per position (DS4F-0252), so warm them
        # all here instead of paying a JIT inside the first request (~1-2 s per shape per layer kind)
        for k in range(c.max_seq_len // c.chunk_size):
            self._pending_ids[id(x)] = (x, torch.zeros(c.chunk_size, dtype=torch.int64))
            self.prefill_chunk(
                x, kv_caches, slot_id=0, actual_start=k * c.chunk_size, actual_end=(k + 1) * c.chunk_size, warmup=True
            )
        for layer in self.model.layers:
            layer.reset_slot(0)
        _clear_export_caches(kv_caches, "the compile warm-up")
        ttnn.synchronize_device(self.mesh_device)
        self._compiled = True
        self._trace_captured = False
        self._log_dram("after compile")
        logger.info(
            f"[v4 runtime] compiled: layers {c.first_layer_idx}..{c.first_layer_idx + c.num_layers - 1}, chunk {c.chunk_size}, max_seq {c.max_seq_len}, users {c.num_users}"
        )

    def capture_trace(self, kv_caches=None) -> None:
        """use_trace: capture every layer's islands ONCE (after compile's eager warm-up). Idempotent; a no-op when
        use_trace is off. The engine calls this after its D2D endpoints and ack channels are wired (their L1 must be
        allocated before anything is captured -- same rule as the MLA runtime)."""
        c = self.config
        if not c.use_trace or getattr(self, "_trace_captured", False):
            return
        assert self._compiled, "capture_trace needs compile() first (the islands' programs must exist)"
        x = self.make_chunk_input([0] * c.chunk_size)
        self._pending_ids.pop(id(x), None)
        # The hash gate's ids live in ONE persistent buffer for the whole prefill: the per-chunk view is copied into it
        # before any island replays, so no eager tensor that a later layer reads survives a replay (the trace's
        # intermediates land wherever DRAM was free at capture time -- a live view there is overwritten).
        self._ids_dev = self._token_ids_view(x) if self._needs_token_ids else None
        # DS4F-0300 (PREFILL_MOE_TRACED_PADDING=1): island B's MoE reads (slot, start, end) from persistent 1-element device
        # tensors, so a ragged chunk's pad rows get the sentinel expert ON DEVICE (not dispatched) instead of the capture's
        # full-chunk config; allocated before any capture (DS4F-0271), refreshed per chunk in prefill_chunk (value-cached)
        self._moe_meta = None
        if os.environ.get("PREFILL_MOE_TRACED_PADDING", "1") == "1":  # default on: bit-exact (DS4F-0300 s5)
            self._moe_meta = tuple(self._meta1_dev(v) for v in (0, 0, c.chunk_size))
            self._moe_meta_vals = [0, 0, c.chunk_size]
        self.model.enable_trace_islands(x, input_ids=self._ids_dev, moe_meta=self._moe_meta)
        self._trace_captured = True
        # warm the traced path once (the islands' copy / reshape programs compile here, not on the first request)
        out = self.prefill_chunk(x, kv_caches, slot_id=0, actual_start=0, actual_end=c.chunk_size, warmup=True)
        if out is not None and os.environ.get("PREFILL_WARM_TAIL_ROWS", "1") == "1":
            # DS4F-0300: tail_hidden_row slices the 32-row tile holding the request's last row, and the tile start is part of the
            # program -> every NEW local tile start compiled on its first request (~300 ms, measured 256 / 1k / 2k vs 4k).
            # Compile all chunk // sp // 32 of them here (transient slices; the to_torch path too).
            s_l = c.chunk_size // c.sp_factor
            t_w = time.perf_counter()
            for row in range(0, s_l, 32):
                self.tail_hidden_row(out, row)
            logger.info(
                f"[v4 runtime] warmed {s_l // 32} tail-row slice programs in {(time.perf_counter() - t_w) * 1e3:.0f} ms"
            )
        _clear_export_caches(kv_caches, "the capture warm chunk")
        for layer in self.model.layers:
            layer.reset_slot(0)
        ttnn.deallocate(x)
        ttnn.synchronize_device(self.mesh_device)
        segs = sum(
            (0 if layer._islands is None else sum(i.num_segments for i in layer._islands[:2] if i is not None))
            for layer in self.model.layers
        )
        logger.info(f"[v4 runtime] trace islands captured: {len(self.model.layers)} layers, {segs} trace segments")
        self._log_dram("after capture_trace")
        self._log_state_addresses()

    def _meta1_dev(self, val: int):
        """One persistent 1-element uint32 replicated-DRAM scalar (the MLA runtime's _meta1_dev)."""
        return ttnn.from_torch(
            torch.tensor([val], dtype=torch.int64).reshape(1, 1, 1, 1),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _set_moe_meta(self, idx: int, val: int) -> None:
        if self._moe_meta_vals[idx] == int(val):
            return
        host = ttnn.from_torch(
            torch.tensor([int(val)], dtype=torch.int64).reshape(1, 1, 1, 1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        ttnn.copy_host_to_device_tensor(host, self._moe_meta[idx])
        self._moe_meta_vals[idx] = int(val)

    def _log_dram(self, tag: str) -> None:
        """Per-bank DRAM occupancy (the chunk-10240 43-layer run OOMed at 4.026 of 4.071 GB per bank, DS4F-0260)."""
        try:
            mv = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.DRAM)
            logger.info(
                f"[v4 runtime] DRAM {tag}: {mv.total_bytes_allocated_per_bank / 2**30:.3f} GB allocated / bank, "
                f"{mv.total_bytes_free_per_bank / 2**30:.3f} GB free, largest block {mv.largest_contiguous_bytes_free_per_bank / 2**20:.0f} MB "
                f"({mv.num_banks} banks)"
            )
        except Exception as e:  # informational only
            logger.info(f"[v4 runtime] DRAM {tag}: memory view unavailable ({type(e).__name__})")

    def _log_state_addresses(self) -> None:
        """PREFILL_DRAM_ADDR_DUMP=1 (DS4F-0272): bank-local DRAM address + shape of every per-slot attention state tensor
        (CSA compressed_kv / index_k / slab_rm / score_mask, HCA compressed_kv / tail, sliding_carry) after the lazies exist,
        so a failure that starts at a fixed row can be checked against an address boundary (2^31, 2^32)."""
        if os.environ.get("PREFILL_DRAM_ADDR_DUMP", "0") != "1":
            return
        names = ("compressed_kv", "index_k", "slab_rm", "score_mask", "sliding_carry", "tail")
        for layer in getattr(self.model, "layers", []):
            for slot, st in sorted(getattr(layer, "states", {}).items()):
                parts = []
                for n in names:
                    t = getattr(st, n, None)
                    if isinstance(t, ttnn.Tensor):
                        try:
                            parts.append(f"{n}@{int(t.buffer_address()):#x} {tuple(t.padded_shape)} {t.dtype}")
                        except Exception as e:  # informational only
                            parts.append(f"{n}: address unavailable ({type(e).__name__})")
                logger.info(f"[v4 addr] layer {layer.layer_idx} slot {slot}: " + "; ".join(parts))

    def release_trace(self) -> None:
        """The engine's shutdown hook (prefill_runner calls ``runtime.release_trace`` before ``close_mesh_device``):
        free every layer's islands and their sub-device managers while the allocator is still alive. Without it a
        traced runner segfaults at exit in BankManager::deallocate_buffer via SubDeviceManager::~SubDeviceManager ->
        MeshTraceBuffer::~MeshTraceBuffer (DS4F-0258, MEASURED on run14 host 30 after a successful 26-chunk request).
        Idempotent; a no-op when no islands were captured."""
        if getattr(self, "_trace_captured", False):
            self.model.release_islands()
            self._trace_captured = False

    def tail_hidden_row(self, tail_out, row: int):
        """The final-norm output at chunk row ``row`` (0-based within the chunk) as a host fp32 ``[hidden]`` vector, plus the
        host ms it took. ``tail_out`` is the transformer tail's output, per chip ``[1, 1, S_l, D_l]``: SP-sharded on rows and
        TP-sharded on the hidden dim over the (sp, tp) mesh; one tile-aligned 32-row slice is pulled from every chip and the
        row's SP shard picked on the host. Disaggregated prefill needs only the request's FIRST TOKEN, so the LM head is
        applied on the host by the consumer of this row (the migration driver), not on the device (DS4F-0263)."""
        c = self.config
        t0 = time.perf_counter()
        s_l = c.chunk_size // c.sp_factor
        sp_idx, local = divmod(int(row), s_l)
        tile0 = (local // 32) * 32
        d_l = int(tail_out.shape[-1])
        sl = ttnn.slice(tail_out, [0, 0, tile0, 0], [1, 1, tile0 + 32, d_l])
        dims = [0, 0]
        dims[c.sp_axis], dims[c.tp_axis] = 2, 3
        host = ttnn.to_torch(
            sl,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, mesh_shape=tuple(c.mesh_shape), dims=tuple(dims)),
        )
        ttnn.deallocate(sl)
        h = host[0, 0, sp_idx * 32 + (local - tile0)].float().clone()  # [hidden]
        return h, (time.perf_counter() - t0) * 1e3

    def make_chunk_input(self, token_ids: list):
        c = self.config
        if c.is_first_rank:
            ids = list(token_ids)
            assert len(ids) == c.chunk_size, (len(ids), c.chunk_size)
            t = prepare_prefill_input_tensor(ids, self.mesh_device, c.sp_factor, False, tuple(c.mesh_shape), c.sp_axis)
            self._pending_ids[id(t)] = (t, torch.tensor(ids, dtype=torch.int64))
            return t
        return self.make_placeholder_activation()

    def _token_ids_from_device(self, input_tensor) -> torch.Tensor:
        c = self.config
        full = ttnn.to_torch(
            input_tensor,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, mesh_shape=tuple(c.mesh_shape), dims=(0, 1)),
        )  # [sp, tp, S/sp]: rows = the SP shards, the TP replicas identical
        return full[:, 0, :].reshape(-1).to(torch.int64)

    def _token_ids_view(self, input_tensor):
        """The chunk's token ids as the hash gate's device layout -- per chip ``[S/sp/32, 32]`` uint32 ROW_MAJOR, i.e.
        the global ``[S/32, 32]`` SP-sharded exactly as ``TtMoEGatePrefill._input_ids_to_device`` lays it out --
        reshaped on device from the H2D input ``[1, 1, S/sp]`` per chip. No host round trip (the read-back +
        re-upload of ``_token_ids_from_device`` was two host writes per chunk, one of them inside the forward, which
        a trace capture cannot hold -- DS4F-0247)."""
        c = self.config
        per_chip = c.chunk_size // c.sp_factor
        assert per_chip % 32 == 0, per_chip
        return ttnn.reshape(input_tensor, (per_chip // 32, 32))

    def make_placeholder_activation(self):
        """A non-first rank's input: the packed 4 streams [1, 1, S_l, 4 D_l] the D2D socket overwrites."""
        c = self.config
        return ttnn.from_torch(
            torch.zeros(1, 1, c.chunk_size // c.sp_factor, 4 * self.hf_config.hidden_size // c.tp_factor),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def prefill_chunk(
        self,
        input_tensor,
        kv_caches,
        *,
        slot_id: int,
        actual_start: int,
        actual_end: int,
        request_id: int = 0,
        d2h_service=None,
        record_dev=None,
        warmup: bool = False,
    ):
        c = self.config
        assert d2h_service is None, "the V4 runtime acks through set_layer_ack_channel / set_layer_completion_sink"
        assert 0 <= slot_id < c.num_users and actual_start <= actual_end <= actual_start + c.chunk_size
        assert actual_start + c.chunk_size <= c.max_seq_len
        ids = None
        if c.is_first_rank:
            entry = self._pending_ids.pop(id(input_tensor), None)
            ids = None if entry is None else entry[1]
            if ids is None and self._needs_token_ids:
                # request mode: the chunk arrived over the H2D socket as a device tensor [sp, 1, S/sp] uint32
                # (SP-sharded, TP-replicated) -- hand the hash-routed MoE layers a device view of it
                ids = self._token_ids_view(input_tensor)
            if ids is not None and getattr(self, "_trace_captured", False) and self._ids_dev is not None:
                if isinstance(ids, torch.Tensor):
                    ids = self.model.layers[0].moe.gate._input_ids_to_device(ids)
                if ids is not self._ids_dev:
                    ttnn.copy(ids, self._ids_dev)
                    ttnn.deallocate(ids)
                    ids = self._ids_dev
        self._request_id = int(request_id)
        # LayerAck granularity (PREFILL_LAYER_ACK_MODE): "layer" drains the device after EVERY layer before acking it
        # (the engine's pipelined per-layer migration; serialises host issue and device work 43 times per chunk),
        # "chunk" (default) lets the layers queue up and drains once after the chunk, then acks every layer in order
        # -- the consumer still sees one ack per layer, all of them after the chunk's KV writes have landed.
        deferred: list = []
        cb = None
        if not warmup and self._ack is not None:
            if self._ack_mode == "layer":
                cb = self._ack_after_drain
            else:
                cb = deferred.append
        if getattr(self, "_moe_meta", None) is not None:
            # DS4F-0300: this chunk's real length for island B's padding-aware MoE (start 0: the block lays the chunk out from
            # SP row 0; slot 0: unused by the padding config) -- host write only when the value changes
            self._set_moe_meta(2, int(actual_end) - int(actual_start))
        t_issue0 = time.perf_counter()
        out = self.model(
            input_tensor,
            slot=slot_id,
            caches=kv_caches,
            actual_start=int(actual_start),
            actual_end=int(actual_end),
            input_ids=ids,
            on_layer_complete=cb,
            on_layer_hidden=self._layer_hidden_hook(int(actual_start)) if not warmup else self._drafter_alloc_hook(),
        )
        t_issue1 = time.perf_counter()
        from models.demos.deepseek_v3_d_p.tt.v4 import block as _blk

        if _blk._ISLAND_TIMING and _blk.ISLAND_TIMES and not warmup:
            # DS4F-0300: per-chunk device+host seconds per (layer kind, step), summed over the layers (sync after each step)
            parts = sorted(_blk.ISLAND_TIMES.items(), key=lambda kv: -kv[1])
            logger.info(
                f"[v4 timing] chunk @{actual_start} real {int(actual_end) - int(actual_start)}: total "
                f"{sum(_blk.ISLAND_TIMES.values()) * 1e3:.1f} ms | "
                + ", ".join(f"{k}:{st} {v * 1e3:.1f}" for (k, st), v in parts)
            )
            _blk.ISLAND_TIMES.clear()
        elif warmup:
            _blk.ISLAND_TIMES.clear()
        if deferred:
            if self._ack_mode == "lag1":
                # LAG-1 acks: mark this chunk's end with a device event and ack the PREVIOUS chunk once its event has passed
                # -- by then this chunk's issue has already overlapped the previous chunk's device tail, so the host never
                # idles on a drain (0.1-0.2 s per chunk on run16, INFERRED from cadence - per-layer device sums). The last
                # chunk's acks are flushed by flush_acks() (runner: at the shutdown sentinel) or by the next chunk.
                ev = ttnn.record_event(self.mesh_device)
                prev = self._pending_ack
                self._pending_ack = (ev, list(deferred), int(actual_start))
                if prev is not None:
                    self._fire_pending(prev, t_issue0, t_issue1)
            else:
                ttnn.synchronize_device(self.mesh_device)
                t_drain = time.perf_counter()
                for layer_idx in deferred:
                    self._ack(layer_idx)
                if not warmup and self._chunk_log_every and (self._chunks_logged % self._chunk_log_every == 0):
                    # host issue vs the device tail the per-chunk drain waits for (both ms); the acks are host-only
                    logger.info(
                        f"[v4 runtime] chunk @{actual_start}: issue {(t_issue1 - t_issue0) * 1e3:.1f} ms, drain wait "
                        f"{(t_drain - t_issue1) * 1e3:.1f} ms, acks {(time.perf_counter() - t_drain) * 1e3:.2f} ms ({len(deferred)} layers)"
                    )
            self._chunks_logged += 1
        if isinstance(ids, ttnn.Tensor) and ids is not getattr(self, "_ids_dev", None):
            ttnn.deallocate(ids)
        if c.is_last_rank:
            # the last rank has nothing to forward; with the tail built its output is the final-norm activation the runner
            # turns into the request's first token (tail_hidden_row, DS4F-0263) -- hand it back, else None
            return out if getattr(self.model, "build_tail", False) else None
        return out

    _chunks_logged = 0
    _chunk_log_every = int(os.environ.get("PREFILL_CHUNK_LOG_EVERY", "5"))  # 0 = never
    # DS4F-0272 probe: PREFILL_HIDDEN_PROBE_STARTS="35840,40960" pulls the residual streams to the host after EVERY layer of
    # the chunks that start at those positions and logs, per stream, the largest finite |x| and where the non-finite
    # values are (chip, first bad row, bad-row count). Off by default (empty); a probed chunk costs ~1 s of host reads per
    # layer on 32 chips, so it is a debugging aid for a runner started with the variable, never a serving default.
    _hidden_probe_starts = frozenset(
        int(x) for x in os.environ.get("PREFILL_HIDDEN_PROBE_STARTS", "").split(",") if x.strip()
    )

    # DS4F-0273 (disaggregated prefill + speculative decode): the ring's DSpark drafter conditions on
    # main_x = main_norm(main_proj(cat(lane means of layers 40..42))) for the anchor and a 128-position window. Those lane
    # means (the mean over the 4 hyper-connection streams of each target layer's output) are ours to export: per chunk,
    # the hook keeps a device copy of each target layer's stream mean; after the chunk the runner pulls the last
    # PREFILL_DRAFTER_WINDOW rows (tile-aligned slices, like tail_hidden_row) and publishes them next to the tail row.
    _drafter_window = int(os.environ.get("PREFILL_DRAFTER_WINDOW", "0"))  # 0 = off
    _drafter_layers = tuple(
        int(x) for x in os.environ.get("PREFILL_DRAFTER_LAYERS", "40,41,42").split(",") if x.strip()
    )
    _lane_means: dict = {}  # layer -> PERSISTENT [1, 1, S_l, D_l] buffer, allocated before any trace capture
    _lane_mean_shard_check: Optional[bool] = None

    def _stream_mean(self, streams):
        ts = [t for t in streams if isinstance(t, ttnn.Tensor)]
        if not ts:
            return None
        acc = ttnn.add(ts[0], ts[1]) if len(ts) > 1 else ttnn.clone(ts[0])
        for t in ts[2:]:
            nxt = ttnn.add(acc, t)
            ttnn.deallocate(acc)
            acc = nxt
        mean = ttnn.multiply(acc, 1.0 / len(ts))
        ttnn.deallocate(acc)
        return mean

    def _drafter_alloc_hook(self):
        """compile()'s eager warm-up (BEFORE any island is captured): allocate the persistent lane-mean buffer of every
        target layer as a clone of its first mean. A buffer allocated after a capture can sit on a replay's intermediate
        addresses and be overwritten (DS4F-0262 class -- launch 14: layer 41's exported window was NaN)."""
        if self._drafter_window <= 0 or all(int(l) in self._lane_means for l in self._drafter_layers):
            return None

        def hook(layer_idx, streams):
            if layer_idx not in self._drafter_layers or int(layer_idx) in self._lane_means:
                return
            if not isinstance(streams, (list, tuple)):
                return
            mean = self._stream_mean(streams)
            if mean is not None:
                self._lane_means[int(layer_idx)] = mean  # persistent from here on
                logger.info(f"[v4 runtime] drafter window: lane-mean buffer for layer {layer_idx} {tuple(mean.shape)}")

        return hook

    def _drafter_hook(self):
        if self._drafter_window <= 0:
            return None

        def hook(layer_idx, streams):
            if layer_idx not in self._drafter_layers or not isinstance(streams, (list, tuple)):
                return
            dst = self._lane_means.get(int(layer_idx))
            mean = self._stream_mean(streams)
            if mean is None:
                return
            if dst is None:  # no pre-capture allocation happened (use_trace off): keep it, nothing replays over it
                self._lane_means[int(layer_idx)] = mean
                return
            ttnn.copy(mean, dst)  # into the pre-capture buffer; the transient is freed before the next replay
            ttnn.deallocate(mean)

        return hook

    def _layer_hidden_hook(self, actual_start: int):
        """The per-layer hook the model calls with each layer's residual streams: the DS4F-0272 probe (when its chunk is
        probed) and/or the drafter-window lane means (when PREFILL_DRAFTER_WINDOW > 0)."""
        probe = self._hidden_probe_hook(actual_start)
        drafter = self._drafter_hook()
        if probe is None:
            return drafter
        if drafter is None:
            return probe

        def both(layer_idx, streams):
            probe(layer_idx, streams)
            drafter(layer_idx, streams)

        both.detail = getattr(probe, "detail", False)
        return both

    def _rows_from_all_chips(self, t, lo_local: int, hi_local: int):
        """Tile-aligned rows [lo_local, hi_local) of a per-chip [1, 1, S_l, D_l] tensor from EVERY chip -> host
        [sp, hi_local - lo_local, hidden] fp32 (the TP shards concatenated on the hidden dim)."""
        c = self.config
        tile0, tile1 = (lo_local // 32) * 32, -(-hi_local // 32) * 32
        d_l = int(t.shape[-1])
        sl = ttnn.slice(t, [0, 0, tile0, 0], [1, 1, tile1, d_l])
        dims = [0, 0]
        dims[c.sp_axis], dims[c.tp_axis] = 2, 3
        host = ttnn.to_torch(
            sl,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, mesh_shape=tuple(c.mesh_shape), dims=tuple(dims)),
        )
        ttnn.deallocate(sl)
        n = tile1 - tile0
        host = host[0, 0].float().view(c.sp_factor, n, -1)  # [sp, n, hidden]
        return host[:, lo_local - tile0 : hi_local - tile0]

    def _rows_from_sp_shard(self, t, sp_idx: int, lo_local: int, hi_local: int):
        """Rows [lo_local, hi_local) of SP shard ``sp_idx`` read from its TP chips only -> host fp32 [rows, hidden]; None
        when the mesh order cannot be trusted (falls back to the all-chips read)."""
        c = self.config
        devs = ttnn.get_device_tensors(t)
        rows_, cols_ = int(c.mesh_shape[0]), int(c.mesh_shape[1])
        if len(devs) != rows_ * cols_:
            return None
        parts = []
        for tp_idx in range(c.tp_factor):
            idx = sp_idx * cols_ + tp_idx if c.sp_axis == 0 else tp_idx * cols_ + sp_idx
            parts.append(ttnn.to_torch(devs[idx])[0, 0, lo_local:hi_local].float())
        return torch.cat(parts, dim=-1)

    def drafter_window_rows(self, real_len: int):
        """The last ``min(PREFILL_DRAFTER_WINDOW, real_len)`` rows of every target layer's stream mean for the chunk just
        computed -> (chunk-local row indices, host fp32 [W, len(layers) * hidden] = cat over layers, host ms)."""
        c = self.config
        t0 = time.perf_counter()
        W = min(int(self._drafter_window), int(real_len))
        lo, hi = int(real_len) - W, int(real_len)
        s_l = c.chunk_size // c.sp_factor
        per_layer = []
        for layer in self._drafter_layers:
            m = self._lane_means.get(int(layer))
            if m is None:
                raise RuntimeError(f"[drafter window] no lane mean for layer {layer} (hook did not run?)")
            pieces = []
            r = lo
            while r < hi:
                sp_idx, local = divmod(r, s_l)
                hi_local = min(s_l, local + (hi - r))
                fast = self._rows_from_sp_shard(m, sp_idx, local, hi_local)
                if fast is not None and self._lane_mean_shard_check is None:
                    # one-time check of the mesh-order assumption against the (slow, unambiguous) all-chips read
                    ref = self._rows_from_all_chips(m, local, hi_local)[sp_idx]
                    ok = bool(torch.allclose(ref, fast, rtol=1e-3, atol=1e-3))
                    self._lane_mean_shard_check = ok
                    logger.info(f"[v4 runtime] drafter window: per-shard read matches the all-chips read: {ok}")
                if fast is None or self._lane_mean_shard_check is False:
                    fast = self._rows_from_all_chips(m, local, hi_local)[sp_idx]
                pieces.append(fast)
                r += hi_local - local
            per_layer.append(torch.cat(pieces, 0))  # [W, hidden]
        lanes = torch.cat(per_layer, dim=-1)  # [W, layers * hidden]
        return list(range(lo, hi)), lanes, (time.perf_counter() - t0) * 1e3

    def _hidden_probe_hook(self, actual_start: int):
        if actual_start not in self._hidden_probe_starts:
            return None

        def hook(layer_idx, streams):
            ts = list(streams) if isinstance(streams, (list, tuple)) else [streams]
            parts = []
            for si, t in enumerate(ts):
                if not isinstance(t, ttnn.Tensor):
                    parts.append(f"s{si}: {type(t).__name__}")
                    continue
                mx, bad_chips, n_bad_total, first = 0.0, 0, 0, None
                for ci, d in enumerate(ttnn.get_device_tensors(t)):
                    x = ttnn.to_torch(d).float().reshape(-1, t.shape[-1])
                    fin = torch.isfinite(x)
                    n_bad = int(x.numel() - int(fin.sum()))
                    if fin.any():
                        mx = max(mx, float(x[fin].abs().max()))
                    if n_bad:
                        bad_chips += 1
                        n_bad_total += n_bad
                        if first is None:
                            bad_rows = torch.nonzero(~fin.all(dim=-1)).flatten()
                            first = (ci, int(bad_rows[0]) if len(bad_rows) else -1, int(len(bad_rows)))
                s = f"s{si}: max|x| {mx:.4g}"
                if bad_chips:
                    s += f" NONFINITE {n_bad_total} on {bad_chips} chip(s), first chip {first[0]} row {first[1]} ({first[2]} bad rows)"
                parts.append(s)
            logger.info(f"[v4 probe] chunk @{actual_start} layer {layer_idx}: " + " | ".join(parts))

        # PREFILL_HIDDEN_PROBE_DETAIL=1: the block also reports the attention output (and the island outputs) per layer
        hook.detail = os.environ.get("PREFILL_HIDDEN_PROBE_DETAIL", "0") == "1"
        return hook

    _pending_ack = None  # lag1: (MeshEvent, [layer ids], chunk start) of the last issued, not yet acked chunk

    @property
    def _ack_mode(self) -> str:
        mode = os.environ.get("PREFILL_LAYER_ACK_MODE", "chunk")
        assert mode in (
            "layer",
            "chunk",
            "lag1",
        ), f"PREFILL_LAYER_ACK_MODE={mode!r}: expected 'layer', 'chunk' or 'lag1'"
        return mode

    def _fire_pending(self, pending, t_issue0=None, t_issue1=None) -> None:
        ev, layers, start = pending
        t0 = time.perf_counter()
        ttnn.event_synchronize(ev)
        t1 = time.perf_counter()
        for layer_idx in layers:
            self._ack(layer_idx)
        if self._chunk_log_every and (self._chunks_logged % self._chunk_log_every == 0) and t_issue0 is not None:
            logger.info(
                f"[v4 runtime] chunk @{start} acked (lag1): next chunk issue {(t_issue1 - t_issue0) * 1e3:.1f} ms, event wait "
                f"{(t1 - t0) * 1e3:.1f} ms, acks {(time.perf_counter() - t1) * 1e3:.2f} ms ({len(layers)} layers)"
            )

    def flush_acks(self) -> None:
        """lag1: ack the last issued chunk now (the runner calls this at the end-of-stream sentinel; also safe any time)."""
        pending, self._pending_ack = self._pending_ack, None
        if pending is not None:
            self._fire_pending(pending)

    def _ack_after_drain(self, layer_idx: int) -> None:
        ttnn.synchronize_device(self.mesh_device)  # the unified-cache writes are done before the ack fires
        self._ack(layer_idx)

    def set_layer_ack_channel(self, channel) -> None:
        self._ack = lambda layer_idx: channel.inject(1)

    def set_layer_completion_sink(self, sink) -> None:
        self._ack = lambda layer_idx: sink(layer_idx, self._request_id)

    # ---- migration hooks -----------------------------------------------------------------------------------------
    def kv_migration_base_address(self, kv_caches) -> int:
        for t in kv_caches.group_tensors(include_pending=False).values():
            return int(t.buffer_address())
        raise RuntimeError("no KV group tensor on this rank")

    def build_kv_chunk_table(
        self, kv_caches, path: str, *, first_layer_idx: int = 0, num_my_layers: Optional[int] = None, stage_layout=None
    ) -> str:
        """COLLECTIVE over the runner's ranks (every rank must call it): the contract's multi-config table."""
        return build_v4_kv_chunk_table(
            mesh_device=self.mesh_device,
            caches=kv_caches,
            hf_config=self.hf_config,
            num_slots=self.config.num_users,
            path=path,
            # DS4F-0267: the destination's table must carry the SAME config count. The blaze decode ring has no pending
            # tensors (DS4F-0242) and registers 4 configs; PREFILL_KV_TABLE_PENDING=0 drops the two pending configs here.
            include_pending=os.environ.get("PREFILL_KV_TABLE_PENDING", "1") == "1",
        )
