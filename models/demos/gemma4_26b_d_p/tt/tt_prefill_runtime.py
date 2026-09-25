# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 single-rank chunked-prefill runtime for the common/prefill engine (and tt-d-gen).

Contract (models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md §2):
  * input: uint32 tokens, per chip ``[1, 1, chunk/sp]`` in DEVICE (block-cyclic) order — tt-d-gen's
    ``ring_sdpa_reshuffle`` / ``make_chunk_input`` already reorder, so no host rotation here.
    Tail padding is ``0xFFFFFFFF`` (tt-d-gen PAD_ID); ``embed_device`` clamps it.
  * ``[actual_start, actual_end)``: KV write offset (32-aligned, need not be chunk-aligned: prefix reuse)
    and end of real tokens (KV beyond it is not written).
  * exactly one layer ack per layer, global order (``set_layer_ack_channel`` / ``set_layer_completion_sink``).
    ``GEMMA4_ACK_SYNC=1`` (default) synchronizes the mesh before each ack so a migration burst never
    reads a layer the device has not finished writing.
  * no logits: the populated KV cache is the output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.demos.common.prefill.adapter import KvCaches
from models.demos.gemma4_26b_d_p.reference.config import FULL, SLIDING, Gemma4TextConfig
from models.demos.gemma4_26b_d_p.reference.weights import CheckpointReader
from models.demos.gemma4_26b_d_p.tt.model import TtGemma4Model


@dataclass
class Gemma4KvCaches(KvCaches):
    """{SLIDING|FULL: Gemma4KVCache}; ``[i]`` indexes (sliding, full) for engine code that expects a list."""

    by_type: dict = field(default_factory=dict)

    def __getitem__(self, i):
        return list(self.by_type.values())[i]


@dataclass
class Gemma4RuntimeConfig:
    num_layers: int
    max_seq_len: int
    chunk_size: int
    mesh_shape: tuple
    num_users: int = 1
    first_layer_idx: int = 0
    is_first_rank: bool = True
    is_last_rank: bool = True
    use_trace: bool = False
    sp_axis: int = 0
    tp_axis: int = 1
    fabric_config: object = None
    ckpt_dir: Optional[str] = None
    expert_dtype: object = ttnn.bfloat8_b

    @property
    def sp_factor(self):
        return self.mesh_shape[self.sp_axis]

    @property
    def tp_factor(self):
        return self.mesh_shape[self.tp_axis]


class Gemma4PrefillRuntime:
    def __init__(self, mesh_device, config: Gemma4RuntimeConfig):
        assert config.is_first_rank and config.is_last_rank and config.first_layer_idx == 0, "gemma4_26b_d_p: single-rank only (no PP yet)"
        self.mesh_device = mesh_device
        self.config = config
        self.cfg = Gemma4TextConfig.from_json(Path(config.ckpt_dir) / "config.json" if config.ckpt_dir else None)
        reader = CheckpointReader(config.ckpt_dir) if config.ckpt_dir else CheckpointReader()
        self.model = TtGemma4Model(
            mesh_device, self.cfg, reader, fabric_config=config.fabric_config or ttnn.get_fabric_config(), max_seq_len=config.max_seq_len,
            chunk_size=config.chunk_size, layers=list(range(config.num_layers)), expert_dtype=config.expert_dtype,
            build_lm_head=False, num_users=config.num_users, allocate_kv=False,
        )
        self._on_layer_complete = None
        self._ack_sync = os.environ.get("GEMMA4_ACK_SYNC", "1") == "1"
        self.compiled = False

    # ---------------------------------------------------------------- caches
    def allocate_kv_caches(self) -> Gemma4KvCaches:
        return Gemma4KvCaches(self.model.allocate_kv_caches(self.config.num_users))

    def _bind(self, kv_caches: Gemma4KvCaches):
        self.model.kv = kv_caches.by_type

    # ---------------------------------------------------------------- inputs
    def make_chunk_input(self, token_ids, chunk_size: Optional[int] = None) -> ttnn.Tensor:
        """``token_ids`` [chunk] in DEVICE order (the H2D layout) -> uint32 [1,1,chunk/sp] per chip."""
        chunk_size = chunk_size or self.config.chunk_size
        assert chunk_size == self.config.chunk_size, "gemma4_26b_d_p serves one chunk size"
        t = torch.as_tensor(token_ids, dtype=torch.int64)
        assert t.numel() == chunk_size, f"chunk input must be exactly {chunk_size} tokens (pad the tail), got {t.numel()}"
        return self.model.tokens_to_device(t)

    # ---------------------------------------------------------------- run
    def compile(self, kv_caches: Gemma4KvCaches) -> None:
        """Warm two chunks (the second exercises the cache-backed ring read) into slot 0."""
        c = self.config.chunk_size
        n = 2 if self.config.max_seq_len >= 2 * c else 1
        for i in range(n):
            self.prefill_chunk(self.make_chunk_input([0] * c), kv_caches, slot_id=0, actual_start=i * c, actual_end=(i + 1) * c)
        ttnn.synchronize_device(self.mesh_device)
        self.compiled = True

    def prefill_chunk(self, input_tensor, kv_caches, *, slot_id: int, actual_start: int, actual_end: int, request_id: int = -1,
                      d2h_service=None, metadata_msg=None, **_unused):
        if d2h_service is not None:
            raise NotImplementedError("gemma4_26b_d_p emits layer acks from the host callback; run with PREFILL_LAYER_ACK_D2H=0")
        c = self.config
        assert 0 <= slot_id < c.num_users, f"slot {slot_id} out of range [0, {c.num_users})"
        assert actual_start % ttnn.TILE_SIZE == 0, f"actual_start {actual_start} must be 32-aligned"
        assert actual_start < actual_end <= actual_start + c.chunk_size <= c.max_seq_len, (actual_start, actual_end)
        if c.sp_factor > 1 and actual_start % c.chunk_size:
            # The ring SDPA sliding-window path needs each chunk to end on a ring-group boundary
            # (ring_joint_sdpa_device_operation.cpp: logical_n % (chunk) == 0). SP=1 (1x4) handles any 32-aligned start.
            raise ValueError(
                f"gemma4_26b_d_p on SP={c.sp_factor}: actual_start={actual_start} must be a multiple of chunk_size={c.chunk_size}. "
                f"In tt-d-gen set runtime.kv_block_size == chunk_size ({c.chunk_size}) so prefix reuse resumes on chunk boundaries."
            )
        self._bind(kv_caches)
        x = self.model.embed_device(input_tensor)
        ttnn.deallocate(input_tensor)
        cb = None
        if self._on_layer_complete is not None:
            def cb(layer_idx, _rid=request_id):
                if self._ack_sync:
                    ttnn.synchronize_device(self.mesh_device)
                self._on_layer_complete(layer_idx, _rid)
        out = self.model.forward_device(x, actual_start, user=slot_id, valid_end=actual_end, on_layer_complete=cb)
        out.deallocate(True)
        return None

    # ---------------------------------------------------------------- acks
    def set_layer_ack_channel(self, layer_ack_channel) -> None:
        self._on_layer_complete = lambda layer_idx, request_id: layer_ack_channel.inject(1)

    def set_layer_completion_sink(self, sink) -> None:
        self._on_layer_complete = lambda layer_idx, request_id: sink(layer_idx, request_id)

    # ---------------------------------------------------------------- migration
    def kv_migration_stages(self, kv_caches: Gemma4KvCaches, first_layer_idx=None, num_my_layers=None):
        from models.demos.common.prefill.runners.migration import KvCacheStage

        n = self.config.num_layers if num_my_layers is None else int(num_my_layers)
        first = 0 if first_layer_idx is None else int(first_layer_idx)
        return [KvCacheStage(int(t.buffer_address()), first, n) for cache in kv_caches.by_type.values() for t in (cache.k, cache.v)]

    def kv_migration_base_address(self, kv_caches: Gemma4KvCaches) -> int:
        return int(kv_caches.by_type[SLIDING].k.buffer_address())

    def build_kv_chunk_table(self, kv_caches: Gemma4KvCaches, path: str, *, first_layer_idx: int = 0, num_my_layers=None, **_layouts) -> str:
        from models.demos.gemma4_26b_d_p.tt.runners.kv_chunk_table import build_and_serialize_kv_chunk_table

        return build_and_serialize_kv_chunk_table(
            path=path, mesh_device=self.mesh_device, cfg=self.cfg, caches=kv_caches.by_type, cache_layer=self.model.cache_layer,
            seq_len=self.config.max_seq_len, chunk_size=self.config.chunk_size, num_users=self.config.num_users,
        )

    # ---------------------------------------------------------------- readback (validation)
    def read_layer_kv(self, kv_caches: Gemma4KvCaches, slot: int, layer: int, n_tokens: int):
        """Device K/V of one GLOBAL layer, natural token order, [1, n_kv, n_tokens, D] each (K in Meta-rope order)."""
        from models.demos.gemma4_26b_d_p.tt.attention.attention import kv_heads_for_col
        from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions

        t = self.cfg.layer_types[layer]
        cache = kv_caches.by_type[t]
        sp, tp = tuple(self.mesh_device.shape)
        n_kv = self.cfg.layer_kv_heads(layer)
        b = slot * cache.num_layers + self.model.cache_layer[layer]
        pos = blockcyclic_positions(sp, self.config.chunk_size, self.config.max_seq_len)

        def heads(tensor):
            dts = ttnn.get_device_tensors(ttnn.slice(tensor, [b, 0, 0, 0], [b + 1, cache.n_kv_local, tensor.shape[2], cache.head_dim],
                                                     memory_config=ttnn.DRAM_MEMORY_CONFIG))
            out = [None] * n_kv
            for c in range(tp):
                idx = kv_heads_for_col(c, tp, self.cfg.num_attention_heads, n_kv)
                dev = torch.cat([ttnn.to_torch(dts[r * tp + c]).float()[0] for r in range(sp)], dim=1)  # [n_kv_local, seq, D]
                nat = torch.empty_like(dev)
                nat[:, pos] = dev
                for hl, g in enumerate(idx):
                    if out[g] is None:
                        out[g] = nat[hl, :n_tokens]
            return torch.stack(out)[None]

        return heads(cache.k), heads(cache.v)
