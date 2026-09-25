# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2 chunked-prefill runtime for the common/prefill engine (and tt-d-gen).

Contract (models/demos/common/prefill/docs/ADDING_A_PREFILL_MODEL.md §2):
  * first rank input: uint32 tokens, per chip ``[1, 1, chunk/sp]`` in DEVICE (block-cyclic) order (tt-d-gen's
    ``ring_sdpa_reshuffle`` already reorders); tail pad ``0xFFFFFFFF`` (PAD_ID) is clamped on device.
    Non-first pipeline ranks take the previous rank's hidden ``[1, 1, chunk/sp, H]`` (D2D socket).
  * ``[actual_start, actual_end)``: KV write offset (32-aligned; chunk-aligned on SWA — the sliding ring
    needs whole ring groups) and end of real tokens (KV beyond it is not written).
  * exactly one layer ack per layer, global order. ``MIMO_ACK_SYNC=1`` (default) synchronizes before each
    ack so a migration burst never reads a layer the device has not finished writing.
  * last rank emits no logits: the populated KV cache is the output; other ranks return their hidden.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import torch

import ttnn
from models.demos.common.prefill.adapter import KvCaches
from models.demos.mimo_v2_d_p.reference.config import GA, SWA, MiMoTextConfig
from models.demos.mimo_v2_d_p.tt.model import TtMiMoModel


@dataclass
class MiMoKvCaches(KvCaches):
    """{GA|SWA: MiMoKVCache}; ``[i]`` indexes in insertion order for engine code that expects a list."""

    by_type: dict = field(default_factory=dict)

    def __getitem__(self, i):
        return list(self.by_type.values())[i]


@dataclass
class MiMoRuntimeConfig:
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
    expert_dtype: object = None  # None -> ffn.default_expert_dtype() (MIMO_EXPERT_DTYPE)

    @property
    def sp_factor(self):
        return self.mesh_shape[self.sp_axis]

    @property
    def tp_factor(self):
        return self.mesh_shape[self.tp_axis]


class MiMoPrefillRuntime:
    def __init__(self, mesh_device, config: MiMoRuntimeConfig, *, layer_state=None, global_state=None):
        from models.demos.mimo_v2_d_p.reference import weights

        self.mesh_device = mesh_device
        self.config = config
        self.cfg = MiMoTextConfig.from_json(os.path.join(config.ckpt_dir, "config.json") if config.ckpt_dir else None)
        layers = list(range(config.first_layer_idx, config.first_layer_idx + config.num_layers))
        self.model = TtMiMoModel(
            mesh_device, self.cfg, layer_state or (lambda i: weights.layer_state(i, self.cfg)), fabric_config=config.fabric_config or ttnn.get_fabric_config(),
            max_seq_len=config.max_seq_len, chunk_size=config.chunk_size, layers=layers, global_state=global_state or weights.global_state,
            num_users=config.num_users, expert_dtype=config.expert_dtype, allocate_kv=False, embed=config.is_first_rank,
        )
        self._on_layer_complete = None
        self._ack_sync = os.environ.get("MIMO_ACK_SYNC", "1") == "1"
        self.compiled = False

    # ---------------------------------------------------------------- caches
    def allocate_kv_caches(self) -> MiMoKvCaches:
        return MiMoKvCaches(self.model.allocate_kv_caches(self.config.num_users))

    def _bind(self, kv_caches: MiMoKvCaches):
        self.model.kv = kv_caches.by_type

    # ---------------------------------------------------------------- inputs
    def make_chunk_input(self, token_ids=None, chunk_size: Optional[int] = None) -> ttnn.Tensor:
        """First rank: ``token_ids`` [chunk] in DEVICE order -> uint32 [1,1,chunk/sp] per chip. Other ranks: a
        placeholder hidden [1,1,chunk/sp,H] the D2D socket fills."""
        c = self.config
        assert (chunk_size or c.chunk_size) == c.chunk_size, "mimo_v2_d_p serves one chunk size"
        if not c.is_first_rank:
            return ttnn.from_torch(
                torch.zeros(1, 1, c.chunk_size, self.cfg.hidden_size), device=self.mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=tuple(c.mesh_shape), dims=(2, None)),
            )
        t = torch.as_tensor(token_ids, dtype=torch.int64)
        assert t.numel() == c.chunk_size, f"chunk input must be exactly {c.chunk_size} tokens (pad the tail), got {t.numel()}"
        return self.model.tokens_to_device(t)

    # ---------------------------------------------------------------- run
    def compile(self, kv_caches: MiMoKvCaches) -> None:
        """Warm two chunks (the second exercises the cache-backed ring read) into slot 0."""
        c = self.config
        n = 2 if c.max_seq_len >= 2 * c.chunk_size else 1
        for i in range(n):
            inp = self.make_chunk_input([0] * c.chunk_size)
            out = self.prefill_chunk(inp, kv_caches, slot_id=0, actual_start=i * c.chunk_size, actual_end=(i + 1) * c.chunk_size)
            if out is not None:
                out.deallocate(True)
        ttnn.synchronize_device(self.mesh_device)
        self.compiled = True

    def prefill_chunk(self, input_tensor, kv_caches, *, slot_id: int, actual_start: int, actual_end: int, request_id: int = -1,
                      d2h_service=None, metadata_msg=None, **_unused):
        if d2h_service is not None:
            raise NotImplementedError("mimo_v2_d_p emits layer acks from the host callback; run with PREFILL_LAYER_ACK_D2H=0")
        c = self.config
        assert 0 <= slot_id < c.num_users, f"slot {slot_id} out of range [0, {c.num_users})"
        assert actual_start % ttnn.TILE_SIZE == 0, f"actual_start {actual_start} must be 32-aligned"
        assert actual_start < actual_end <= actual_start + c.chunk_size <= c.max_seq_len, (actual_start, actual_end)
        if actual_start % c.chunk_size and self.model.layer_counts[SWA]:
            raise ValueError(
                f"mimo_v2_d_p: actual_start={actual_start} must be a multiple of chunk_size={c.chunk_size} (the SWA ring needs whole "
                f"ring groups). In tt-d-gen set runtime.kv_block_size == chunk_size so prefix reuse resumes on chunk boundaries."
            )
        self._bind(kv_caches)
        x = self.model.embed_device(input_tensor) if c.is_first_rank else input_tensor
        if c.is_first_rank:
            ttnn.deallocate(input_tensor)
        cb = None
        if self._on_layer_complete is not None:

            def cb(layer_idx, _rid=request_id):
                if self._ack_sync:
                    ttnn.synchronize_device(self.mesh_device)
                self._on_layer_complete(layer_idx, _rid)

        out = self.model.forward_device(x, actual_start, user=slot_id, valid_end=actual_end, on_layer_complete=cb)
        if c.is_last_rank:
            out.deallocate(True)
            return None
        return out

    # ---------------------------------------------------------------- acks
    def set_layer_ack_channel(self, layer_ack_channel) -> None:
        self._on_layer_complete = lambda layer_idx, request_id: layer_ack_channel.inject(1)

    def set_layer_completion_sink(self, sink) -> None:
        self._on_layer_complete = lambda layer_idx, request_id: sink(layer_idx, request_id)

    # ---------------------------------------------------------------- migration
    def kv_migration_stages(self, kv_caches: MiMoKvCaches, first_layer_idx=None, num_my_layers=None):
        from models.demos.common.prefill.runners.migration import KvCacheStage

        n = self.config.num_layers if num_my_layers is None else int(num_my_layers)
        first = self.config.first_layer_idx if first_layer_idx is None else int(first_layer_idx)
        return [KvCacheStage(int(t.buffer_address()), first, n) for cache in kv_caches.by_type.values() for t in (cache.k, cache.v)]

    def kv_migration_base_address(self, kv_caches: MiMoKvCaches) -> int:
        return int(next(iter(kv_caches.by_type.values())).k.buffer_address())

    def build_kv_chunk_table(self, kv_caches: MiMoKvCaches, path: str, **_layouts) -> str:
        from models.demos.mimo_v2_d_p.tt.runners.kv_chunk_table import build_and_serialize_kv_chunk_table

        return build_and_serialize_kv_chunk_table(
            path=path, mesh_device=self.mesh_device, cfg=self.cfg, caches=kv_caches.by_type, cache_layer=self.model.cache_layer,
            seq_len=self.config.max_seq_len, chunk_size=self.config.chunk_size, num_users=self.config.num_users,
        )

    # ---------------------------------------------------------------- readback (validation)
    def read_layer_kv(self, kv_caches: MiMoKvCaches, slot: int, layer: int, n_tokens: int):
        """Device K/V of one GLOBAL layer, natural token order: K [1, n_kv, n, 192] (Meta-rope order), V [1, n_kv, n, 128]."""
        from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
        from models.demos.mimo_v2_d_p.tt.attention.attention import kv_heads_for_col

        spec = self.cfg.layer_attn(layer)
        cache = kv_caches.by_type[spec.kind]
        sp, tp = tuple(self.mesh_device.shape)
        b = slot * cache.num_layers + self.model.cache_layer[layer]
        pos = blockcyclic_positions(sp, self.config.chunk_size, self.config.max_seq_len)

        def heads(tensor, d_keep):
            d = tensor.shape[3]
            dts = ttnn.get_device_tensors(ttnn.slice(tensor, [b, 0, 0, 0], [b + 1, cache.n_kv_local, tensor.shape[2], d],
                                                     memory_config=ttnn.DRAM_MEMORY_CONFIG))
            out = [None] * spec.n_kv
            for c in range(tp):
                idx = kv_heads_for_col(c, tp, spec.n_q, spec.n_kv)
                dev = torch.cat([ttnn.to_torch(dts[r * tp + c]).float()[0] for r in range(sp)], dim=1)
                nat = torch.empty_like(dev)
                nat[:, pos] = dev
                for hl, g in enumerate(idx):
                    if out[g] is None:
                        out[g] = nat[hl, :n_tokens, :d_keep]
            return torch.stack(out)[None]

        return heads(cache.k, spec.head_dim), heads(cache.v, spec.v_head_dim)
