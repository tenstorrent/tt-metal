# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefill runtime (P2): ``compile`` / ``make_chunk_input`` / ``prefill_chunk``.

Lifecycle: build the model once, ``compile(kv_cache)`` to warm every KV-length bucket a served loop
can reach, then ``prefill_chunk`` once per chunk in order. The runtime is **stateless with respect
to the KV cache** — the caller allocates it and passes it into every call that touches it — which is
the contract ``models/demos/common/prefill``'s engine expects and is also what lets a test run two
independent prefills against one runtime.

Input convention: ``make_chunk_input`` returns the chunk's token ids as an SP-sharded uint32 tensor,
row ``r`` holding the contiguous slice ``[r*s_local, (r+1)*s_local)``. That is the same per-chip
layout a serving H2D socket would deliver, so the standalone path and the served path would embed
the same tensor. The embedding happens on device inside ``prefill_chunk``.

RoPE is the whole-cache **indexed** table, built once in ``__init__`` and reused for every chunk; the
op derives each chunk's start row on device from ``kv_actual_global``. Chunk 0 and chunk N therefore
run the same code with the same tensors — the only difference is ``cached_len``, which selects the
live vs cache-read ring SDPA inside attention.

**Chunk-range contract.** ``[actual_start, actual_end)`` is the absolute KV-position range of this
chunk's real tokens: ``actual_start`` is the cache write offset and ``actual_end`` is past the last
real token (a ragged final chunk is shorter, and causality makes its pad tail inert). Every part of
that is asserted — out of range, out of order, past the cache, or straddling more than one chunk —
because an out-of-contract chunk otherwise writes at a wrong offset and silently corrupts the cache
rather than failing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
from loguru import logger

import ttnn

from .attention.kv_cache import allocate_kv_cache, read_slot_kv
from .ccl import CCLManager
from .mesh import MeshConfig
from .model import Model
from ..utils.general import default_num_links


@dataclass
class PrefillRuntimeConfig:
    num_layers: int
    max_seq_len: int  # per-user KV-cache capacity in tokens; a multiple of chunk_size
    chunk_size: int  # tokens per prefill_chunk(); one-shot sets this == max_seq_len
    mesh_shape: tuple = (8, 4)
    num_users: int = 1
    sp_axis: int = 0
    tp_axis: int = 1
    topology: ttnn.Topology = ttnn.Topology.Linear
    cache_dtype: ttnn.DataType = ttnn.bfloat8_b
    weight_dtype: ttnn.DataType = ttnn.bfloat8_b
    weight_cache_path: Optional[str] = None
    shard_vocab_on_sp: Optional[bool] = None
    build_lm_head: bool = False
    num_links: Optional[int] = None
    extra: dict = field(default_factory=dict)

    @property
    def sp(self) -> int:
        return self.mesh_shape[self.sp_axis]

    @property
    def tp(self) -> int:
        return self.mesh_shape[self.tp_axis]


class TtPrefillRuntime:
    def __init__(self, mesh_device, cfg, state_dict: Optional[dict], config: PrefillRuntimeConfig):
        assert config.max_seq_len % config.chunk_size == 0, (
            f"max_seq_len ({config.max_seq_len}) must be a multiple of chunk_size ({config.chunk_size}); "
            f"a partial trailing chunk would write past a block-cyclic period boundary"
        )
        assert config.chunk_size % (ttnn.TILE_SIZE * config.sp) == 0, (
            f"chunk_size ({config.chunk_size}) must be a multiple of TILE_SIZE*sp "
            f"({ttnn.TILE_SIZE * config.sp}) — it is the block-cyclic addressing period of the KV table"
        )
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.config = config
        self.compiled = False
        self._on_layer_complete = None

        self.mesh_config = MeshConfig(config.mesh_shape, tp=config.tp, tp_axis=config.tp_axis)
        self.ccl_manager = CCLManager(
            mesh_device,
            num_links=config.num_links or default_num_links(mesh_device),
            topology=config.topology,
        )
        logger.info(
            f"building prefill runtime: layers={config.num_layers} max_seq_len={config.max_seq_len} "
            f"chunk={config.chunk_size} mesh={config.mesh_shape} users={config.num_users}"
        )
        self.model = Model(
            mesh_device,
            cfg,
            mesh_config=self.mesh_config,
            ccl_manager=self.ccl_manager,
            state_dict=state_dict,
            max_seq_len=config.max_seq_len,
            num_layers=config.num_layers,
            weight_dtype=config.weight_dtype,
            tensor_cache_path=config.weight_cache_path,
            sequence_parallel=True,
            shard_vocab_on_sp=config.shard_vocab_on_sp,
            build_lm_head=config.build_lm_head,
        )
        # Built once; the indexed rope op picks each chunk's rows on device. Persistent — never freed
        # between chunks.
        self.rope_indexed = self.model.rope_setup.build_indexed_rope(config.max_seq_len, config.chunk_size)

    # --- cache ------------------------------------------------------------------------------
    def allocate_kv_cache(self):
        return allocate_kv_cache(
            self.mesh_device,
            num_layers=self.config.num_layers,
            max_seq_len=self.config.max_seq_len,
            num_kv_heads=self.cfg.num_key_value_heads,
            tp=self.mesh_config.tp,
            sp_axis=self.mesh_config.sp_axis,
            num_users=self.config.num_users,
            head_dim=self.cfg.head_dim,
            cache_dtype=self.config.cache_dtype,
        )

    # --- inputs -----------------------------------------------------------------------------
    def make_chunk_input(self, token_ids) -> ttnn.Tensor:
        """One chunk's token ids as an SP-sharded uint32 ROW_MAJOR tensor, per-chip ``(1, 1, s_local)``.

        Row ``r`` holds the contiguous slice ``[r*s_local, (r+1)*s_local)`` of the chunk, replicated
        across the TP columns. Exactly ``chunk_size`` ids: a short final chunk is the caller's to pad,
        and ``actual_end`` is how it says where the real tokens stop.
        """
        assert len(token_ids) == self.config.chunk_size, (
            f"chunk input must be exactly chunk_size={self.config.chunk_size} tokens (pad the tail), "
            f"got {len(token_ids)}"
        )
        sp = self.config.sp
        s_local = self.config.chunk_size // sp
        tok = torch.tensor(token_ids, dtype=torch.int32).reshape(sp, 1, s_local)
        return ttnn.from_torch(
            tok,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, mesh_shape=self.config.mesh_shape, dims=(self.config.sp_axis, None)
            ),
        )

    # --- run --------------------------------------------------------------------------------
    def compile(self, kv_cache) -> None:
        """Warm every KV-length bucket the served loop can reach, so no real chunk pays a first-run JIT.

        Each warm-up writes slot 0 with zero tokens; the real run overwrites those positions, so the
        cache is not left dirty for the range it will fill. A caller that warms and then prefills a
        SHORTER sequence must re-allocate the cache — ``kv_cache_is_dirty_beyond`` says where.
        """
        chunk = self.config.chunk_size
        starts = list(range(0, self.config.max_seq_len - chunk + 1, chunk))
        logger.info(f"compile(): warming {len(starts)} KV-length buckets of {chunk} tokens")
        t0 = time.perf_counter()
        for start in starts:
            self.prefill_chunk(
                self.make_chunk_input([0] * chunk),
                kv_cache,
                slot_id=0,
                actual_start=start,
                actual_end=start + chunk,
            )
        ttnn.synchronize_device(self.mesh_device)
        logger.info(f"compile() warmed {len(starts)} buckets in {(time.perf_counter() - t0) * 1000:.0f} ms")
        self.compiled = True

    def prefill_chunk(
        self,
        input_tensor: ttnn.Tensor,
        kv_cache,
        *,
        slot_id: int = 0,
        actual_start: int,
        actual_end: int,
        skip_lm_head: bool = True,
    ):
        """Prefill ONE chunk into user ``slot_id``'s slice of ``kv_cache``.

        Returns ``None`` under ``skip_lm_head`` (the populated cache is the output) and the
        vocab-sharded logits otherwise. Call once per chunk, in order: a chunk's KV must be in the
        cache before the next chunk reads it.
        """
        c = self.config
        assert 0 <= slot_id < c.num_users, f"slot_id {slot_id} out of range [0, {c.num_users})"
        assert actual_start % c.chunk_size == 0, (
            f"actual_start ({actual_start}) must be a whole number of chunks: it is the cache write "
            f"offset and the block-cyclic writer places rows relative to a chunk boundary"
        )
        assert actual_start + c.chunk_size <= c.max_seq_len, (
            f"chunk at actual_start={actual_start} would run past the per-user cache ({c.max_seq_len})"
        )
        assert actual_start < actual_end <= actual_start + c.chunk_size, (
            f"[actual_start={actual_start}, actual_end={actual_end}) is not within one chunk of {c.chunk_size}"
        )

        x = self.model.embedding(input_tensor)
        if len(x.shape) == 3:
            x = ttnn.unsqueeze_to_4D(x)
        ttnn.deallocate(input_tensor)

        out = self.model.prefill_forward(
            x,
            rope_mats=self.rope_indexed,
            kv_cache=kv_cache,
            cached_len=actual_start,
            user_id=slot_id,
            indexed_rope=True,
            skip_lm_head=skip_lm_head,
            on_layer_complete=self._on_layer_complete,
        )
        if skip_lm_head:
            if out is not None:
                out.deallocate(True)
            return None
        return out

    def prefill_sequence(self, token_ids, kv_cache, *, slot_id: int = 0):
        """Drive a whole prompt through ``prefill_chunk``, one chunk at a time.

        The last chunk is zero-padded to ``chunk_size`` and ``actual_end`` marks where the real
        tokens stop. Returns the number of real tokens written.
        """
        c = self.config
        n = len(token_ids)
        assert n <= c.max_seq_len, f"{n} tokens exceeds the cache capacity {c.max_seq_len}"
        for start in range(0, n, c.chunk_size):
            piece = list(token_ids[start : start + c.chunk_size])
            end = start + len(piece)
            if len(piece) < c.chunk_size:
                piece = piece + [0] * (c.chunk_size - len(piece))
            self.prefill_chunk(
                self.make_chunk_input(piece), kv_cache, slot_id=slot_id, actual_start=start, actual_end=end
            )
        return n

    def set_layer_ack_channel(self, layer_ack_channel) -> None:
        """Per-layer completion seam for a serving engine. Unused here (serving is a follow-on)."""
        assert self.compiled, "call compile() before set_layer_ack_channel()"
        self._on_layer_complete = lambda layer_idx: layer_ack_channel.inject(1)

    # --- read-back --------------------------------------------------------------------------
    def read_slot_kv(self, kv_cache, slot_id: int = 0):
        """Raw (block-cyclic) host copy of one slot's K and V — see ``attention/kv_cache.read_slot_kv``."""
        return read_slot_kv(self.mesh_device, kv_cache, slot_id, self.config.num_layers)
