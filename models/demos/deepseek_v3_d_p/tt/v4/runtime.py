# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``TtV4PrefillRuntime``: the engine-facing runtime of the DeepSeek-V4-Flash prefill (the structural contract of
``models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md`` section 2).

The engine owns the caches (``V4FlashKvCaches``), the chunk loop, the sockets and the migration comms; this runtime
owns the model and knows how to write the caches. Eager only (no trace) in this milestone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
from models.demos.deepseek_v3_d_p.tt.v4.kv_table import build_v4_kv_chunk_table
from models.demos.deepseek_v3_d_p.tt.v4.transformer import TtV4PrefillTransformer


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
        ttnn.synchronize_device(self.mesh_device)
        self._compiled = True
        self._trace_captured = False
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
        self.model.enable_trace_islands(x, input_ids=self._ids_dev)
        self._trace_captured = True
        # warm the traced path once (the islands' copy / reshape programs compile here, not on the first request)
        self.prefill_chunk(x, kv_caches, slot_id=0, actual_start=0, actual_end=c.chunk_size, warmup=True)
        for layer in self.model.layers:
            layer.reset_slot(0)
        ttnn.deallocate(x)
        ttnn.synchronize_device(self.mesh_device)
        segs = sum(
            (0 if layer._islands is None else sum(i.num_segments for i in layer._islands[:2] if i is not None))
            for layer in self.model.layers
        )
        logger.info(f"[v4 runtime] trace islands captured: {len(self.model.layers)} layers, {segs} trace segments")

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
        out = self.model(
            input_tensor,
            slot=slot_id,
            caches=kv_caches,
            actual_start=int(actual_start),
            actual_end=int(actual_end),
            input_ids=ids,
            on_layer_complete=None if warmup else self._ack,
        )
        if isinstance(ids, ttnn.Tensor) and ids is not getattr(self, "_ids_dev", None):
            ttnn.deallocate(ids)
        if c.is_last_rank:
            return None
        return out

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
        )
