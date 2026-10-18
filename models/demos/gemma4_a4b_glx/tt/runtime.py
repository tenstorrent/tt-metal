# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The prefill runner's runtime for gemma4_a4b_glx: one chunk at a time into the engine-owned KV slots.

Input: the H2D chunk, [SP, 1, chunk / SP] uint32 (row r of the mesh holds slice r), in the engine's ring-SDPA order
(ring_sdpa_reshuffle.hpp: a chunk starting at actual_start puts its first chunk/SP - intra tokens on SP chip
(start // w) % SP, w = chunk / SP, intra = start % w). The hidden state is replicated, so the slices are gathered
over the mesh rows and put back in natural order on the device (two slices and a concat), then embedded.

Acks: one per layer, after that layer's K/V writes. With the D2H service the record is a device op enqueued after
the layer (ordered behind its cache writes); with a host sink the device is synchronized before the call.
"""

from __future__ import annotations

import ttnn
from models.demos.common.prefill.adapter import KvCaches

from .attention import TILE, TtKVCache

SP_AXIS = 0


class Gemma4A4bGlxKvCaches(KvCaches):
    """{layer: TtKVCache}, layers 0..n-1 of this rank: [slots, Hkv/TP, max_seq, D] bfp8 per chip, served layout."""

    def __init__(self, mesh, cfg, layers, num_slots: int, max_seq: int):
        self.num_slots, self.max_seq = num_slots, max_seq
        tp = mesh.shape[1]
        self.layers = {}
        for i in layers:
            hkv, d = cfg.attn_dims(i)
            # A global KV head sits on 2 adjacent columns (tt.attention.TtGlobalAttention): TP heads over the mesh.
            heads = hkv if cfg.is_sliding(i) else tp
            self.layers[i] = TtKVCache(mesh, heads, d, max_seq, num_slots)

    def __getitem__(self, layer):
        return self.layers[layer]

    def __len__(self):
        return len(self.layers)

    def tensors(self):
        return [t for i in sorted(self.layers) for t in (self.layers[i].k, self.layers[i].v)]


class Gemma4A4bGlxPrefillRuntime:
    def __init__(self, *, mesh_device, model_path: str, config):
        self.mesh_device = mesh_device
        self.model_path = model_path
        self.config = config
        self.model = None
        self.d2h_service = None
        self.layer_completion_sink = None
        self.sp = int(mesh_device.shape[SP_AXIS])
        assert config.chunk_size % (self.sp * TILE) == 0, f"chunk {config.chunk_size} over SP={self.sp}"

    # ---- setup
    def compile(self, kv):
        """Load the model (this rank's layers) and its RoPE / page tables for the cache length."""
        from .model import TtGemma4Model

        self._check_cache(kv)
        if self.model is None:
            first, n = self.config.first_layer_idx, self.config.num_layers
            self.model = TtGemma4Model(self.mesh_device, self.model_path, layers=range(first, first + n))
        self.model.ensure_tables(self.config.max_seq_len)
        ttnn.synchronize_device(self.mesh_device)

    def set_d2h_ack_service(self, service):
        self.d2h_service = service

    def set_layer_completion_sink(self, sink):
        self.layer_completion_sink = sink

    def warmup_ack_count(self):
        return 0

    def release_trace(self):
        self.d2h_service = None
        self.layer_completion_sink = None

    def _check_cache(self, kv):
        if not isinstance(kv, Gemma4A4bGlxKvCaches):
            raise TypeError(f"expected Gemma4A4bGlxKvCaches, got {type(kv).__name__}")
        if len(kv) != self.config.num_layers or kv.max_seq != self.config.max_seq_len:
            raise ValueError(f"KV caches of {len(kv)} layers x {kv.max_seq} do not match the runtime's")

    # ---- migration
    def kv_migration_stages(self, kv, first_layer_idx, num_my_layers):
        from models.demos.common.prefill.runners.migration import KvCacheStage

        self._check_cache(kv)
        if first_layer_idx != 0 or num_my_layers != self.config.num_layers:
            raise ValueError("gemma4_a4b_glx migration requires all layers on one rank")
        return [
            KvCacheStage(int(t.buffer_address()), layer, 1)
            for layer in sorted(kv.layers)
            for t in (kv[layer].k, kv[layer].v)
        ]

    def kv_migration_base_address(self, kv):
        """Layer 0's K buffer: table entry (layer 0, pos 0, slot 0, config 0) is its first record in bank 0."""
        self._check_cache(kv)
        return int(kv[0].k.buffer_address())

    def build_kv_chunk_table(self, kv, path, *, first_layer_idx=0, num_my_layers=None, stage_layouts=None):
        from .kv_chunk_table import export_table

        stages = self.kv_migration_stages(
            kv, first_layer_idx, self.config.num_layers if num_my_layers is None else num_my_layers
        )
        if stage_layouts is not None:
            if len(stage_layouts) != len(stages):
                raise ValueError(f"{len(stage_layouts)} gathered stage layouts for {len(stages)} cache tensors")
            for gathered, stage in zip(stage_layouts, stages):
                want = {"rank": 0, "base_addr": stage.base_addr, "first_layer": stage.first_layer, "count": stage.count}
                if len(gathered) != 1 or any(gathered[0][k] != v for k, v in want.items()):
                    raise ValueError(f"gathered stage {gathered} does not match the single-rank cache {stage}")
        return export_table(self.mesh_device, kv.layers, kv.num_slots, kv.max_seq, path)

    # ---- serving
    def _tokens(self, input_tensor, start: int):
        """[SP, 1, w] engine-order uint32 (row r: slice r) -> [1, 1, 1, chunk] natural order, replicated."""
        chunk = self.config.chunk_size
        w = chunk // self.sp
        x = ttnn.reshape(input_tensor, (1, 1, 1, w))  # may share the input's buffer: the caller frees the input
        flat = ttnn.all_gather(x, dim=3, cluster_axis=SP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        cs, intra = (start // w) % self.sp, start % w
        if cs == 0 and intra == 0:
            return flat
        # X = flat rotated to start at chip cs; natural = X[0:w-intra] | X[w:chunk] | X[w-intra:w].
        cut = cs * w
        pieces = [(cut + a, cut + b) for a, b in ((0, w - intra), (w, chunk), (w - intra, w))]
        parts = []
        for a, b in pieces:
            for lo, hi in ((a, min(b, chunk)), (max(a, chunk) - chunk, b - chunk)):
                if hi > lo:
                    parts.append(ttnn.slice(flat, [0, 0, 0, lo], [1, 1, 1, hi]))
        ttnn.deallocate(flat)
        out = ttnn.concat(parts, dim=3)
        for p in parts:
            ttnn.deallocate(p)
        return out

    def _validate(self, slot_id, actual_start, actual_end):
        cfg = self.config
        if not 0 <= slot_id < cfg.num_users:
            raise ValueError(f"KV slot {slot_id} outside [0, {cfg.num_users})")
        if actual_start < 0 or actual_start % TILE:
            raise ValueError(f"chunk start {actual_start} is not a nonnegative multiple of {TILE}")
        if not actual_start < actual_end <= actual_start + cfg.chunk_size:
            raise ValueError(f"chunk [{actual_start}, {actual_end}) holds 0 or more than {cfg.chunk_size} tokens")
        if actual_start + cfg.chunk_size > cfg.max_seq_len:
            raise ValueError(f"chunk at {actual_start} runs past the {cfg.max_seq_len}-token cache")

    def prefill_chunk(
        self,
        input_tensor,
        kv,
        *,
        slot_id,
        actual_start,
        actual_end,
        request_id=0,
        d2h_service=None,
        metadata_msg=None,
    ):
        if self.model is None:
            raise RuntimeError("compile(kv) must run before serving chunks")
        self._validate(slot_id, actual_start, actual_end)
        self._check_cache(kv)
        d2h = d2h_service if d2h_service is not None else self.d2h_service
        if d2h is not None and metadata_msg is None:
            raise ValueError("D2H layer acks need the chunk's metadata message")
        ids = self._tokens(input_tensor, actual_start)
        h = self.model.embed(ids)
        ttnn.deallocate(ids)
        for k, block in enumerate(self.model.blocks):
            cache = kv[k]
            cache.slot = slot_id
            h2 = block(h, actual_start, cache, end=actual_end)
            ttnn.deallocate(h)
            h = h2
            if d2h is not None:
                ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(d2h, metadata=metadata_msg)
            elif self.layer_completion_sink is not None:
                ttnn.synchronize_device(self.mesh_device)
                self.layer_completion_sink(k, request_id)
        ttnn.deallocate(h)
        ttnn.deallocate(input_tensor)
        if metadata_msg is not None:
            ttnn.deallocate(metadata_msg)
        return None
