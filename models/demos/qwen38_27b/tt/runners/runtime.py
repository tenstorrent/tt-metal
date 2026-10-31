# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The prefill runtime of Qwen3.8-27B: one chunk at a time from the engine's H2D payload into the engine-owned KV
cache (adapter.py), eager. See adapter.py's docstring for the layout and the ack rule."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import ttnn

from ..model import TtQwen38Model
from . import kv_table
from .adapter import cache_max_seq, served_layers


class Qwen38PrefillRuntime:
    def __init__(self, *, mesh_device, hf_dir: str, config):
        self.mesh_device, self.config = mesh_device, config
        self.layers = served_layers(config)
        # the device model reads hf_dir and target {seq, chunk} off the spec
        spec = SimpleNamespace(
            hf_dir=Path(hf_dir),
            get=lambda key: {"seq": cache_max_seq(config), "chunk": config.chunk_size} if key == "target" else None,
        )
        self.model = TtQwen38Model(mesh_device, spec, self.layers, lm_head=False)
        self.state = None
        self.kv = None
        self.sink = None

    # ---- set-up
    def compile(self, kv) -> None:
        """Bind the engine-owned cache: the model state runs on it (the GDN state is the runtime's own). The first
        chunk compiles the programs (no warm-up chunk: it would write a slot's cache)."""
        if self.state is not None:
            self.state.kv = None  # the engine owns the cache
            self.state.free()
        self.kv = kv
        self.state = self.model.new_state(self.config.max_seq_len, num_slots=self.config.num_users, kv=kv.cache)

    def state_for(self, kv):
        if kv is not self.kv:
            raise ValueError("the runtime runs on the cache given to compile")
        return self.state

    def set_layer_completion_sink(self, sink) -> None:
        self.sink = sink

    # ---- serving
    def prefill_chunk(self, input_tensor, kv, *, slot_id, actual_start, actual_end, request_id=0, **kw):
        """input_tensor: the engine's [sp, 1, chunk / sp] uint32 ids (SP rows over axis 0 in the engine's chip order for
        actual_start, replicated over TP, PAD_ID past actual_end). Writes slot_id's KV rows [actual_start, actual_end)
        and GDN state; acks each served layer once its state is in DRAM. Returns None (single rank)."""
        cfg = self.config
        state = self.state_for(kv)
        if not 0 <= slot_id < cfg.num_users:
            raise ValueError(f"slot {slot_id} outside [0, {cfg.num_users})")
        if actual_start % kv_table.RECORD or not actual_start < actual_end <= actual_start + cfg.chunk_size:
            raise ValueError(
                f"chunk [{actual_start}, {actual_end}): start must be 32-aligned, 1..{cfg.chunk_size} tokens"
            )
        if actual_end > cfg.max_seq_len:
            raise ValueError(f"chunk end {actual_end} past max_seq_len {cfg.max_seq_len}")
        h = self.model.embed_device(input_tensor)  # [1, 1, chunk / sp] per chip, as embed() lays it out
        for i in self.layers:
            out = self.model.layer(i, h, actual_start, state, end=actual_end, slot=slot_id)
            ttnn.deallocate(h)
            h = out
            if self.sink is not None:
                # the layer's KV rows / GDN state and snapshots are final in DRAM once the queue passes this point
                ttnn.event_synchronize(ttnn.record_event(self.mesh_device))
                self.sink(i, request_id)
        ttnn.deallocate(h)
        ttnn.deallocate(input_tensor)
        if kw.get("metadata_msg") is not None:
            ttnn.deallocate(kw["metadata_msg"])
        return None

    # ---- the KV table (serving_contract.md "Adapter and table")
    def kv_migration_stages(self, kv, first_layer_idx, num_my_layers):
        from models.demos.common.prefill.runners.migration import KvCacheStage

        if kv.cache is None:
            return []
        return [KvCacheStage(int(kv.cache.k.buffer_address()), first_layer_idx, num_my_layers)]

    def kv_migration_base_address(self, kv) -> int:
        """Low 32 bits of table entry (layer 0, pos 0, slot 0, config 0), from the cache layout. 0 when served layer
        0 is a Gated DeltaNet layer (its entries are unset); the cache's base is kv_migration_stages()'s."""
        if kv.cache is None:
            return 0
        banks = self.mesh_device.dram_grid_size().x
        return kv_table.entry_noc_addr(kv.cache, kv.attn_of_table_layer(), 0, 0, 0, 0, banks) & 0xFFFFFFFF

    def build_kv_chunk_table(self, kv, path, *, first_layer_idx=0, num_my_layers=None, stage_layouts=None):
        from models.demos.common.prefill.runners.migration import serialize_prebuilt_kv_chunk_table

        n = self.config.num_layers if num_my_layers is None else num_my_layers
        if (first_layer_idx, n) != (kv.first_layer, kv.num_layers):
            raise ValueError(
                f"table layers [{first_layer_idx}, +{n}) != the cache's [{kv.first_layer}, +{kv.num_layers})"
            )
        if kv.cache is None:
            raise ValueError("no attention layer served: no KV records to describe")
        if stage_layouts is not None:
            base = int(kv.cache.k.buffer_address())
            for layout in stage_layouts:
                if len(layout) != 1 or layout[0]["base_addr"] != base or layout[0]["rank"] != 0:
                    raise ValueError(f"single-rank table: stage layout {layout} does not match the cache")
        table = kv_table.build_kv_chunk_table(self.mesh_device, kv.cache, kv.attn_of_table_layer(), n)
        return serialize_prebuilt_kv_chunk_table(table=table, path=str(path))
