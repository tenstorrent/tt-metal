# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The prefill runner's runtime for Qwen3.8-27B: the all-device model (tt/model.py) on the engine-owned serving state
(tt/runners/kv_cache.py), one chunk at a time, eager (no trace).

Per chunk: the engine's H2D tokens -> embedding -> every served layer on the slot's state -> one ack per layer, linear
layers included (the runner sizes the ring for every layer and the engine retires a chunk after layers_per_chunk acks).
An ack follows its layer's ops on the device queue: with the D2H service it is a device record enqueued after the
layer (the layer's KV writes and committed conv / recurrent state land first); with a host sink the runtime
synchronizes the device before the callback.
"""

from __future__ import annotations

import ttnn

from . import kv_cache as K


class Qwen38PrefillRuntime:
    def __init__(self, *, mesh_device, hf_dir: str, config):
        self.mesh_device = mesh_device
        self.hf_dir = hf_dir
        self.config = config
        self.model = None
        self.state = None
        self._sink = None
        self._d2h_service = None

    @property
    def layers(self) -> list[int]:
        first = self.config.first_layer_idx
        return list(range(first, first + self.config.num_layers))

    # ---- build
    def compile(self, kv_caches: K.Qwen38KvCaches) -> None:
        """Load the served layers' weights and bind the model's state to the engine-owned cache."""
        from models.demos.qwen38_27b.tt.model import TtModel

        cfg = self.config
        assert cfg.is_first_rank and cfg.is_last_rank, "Qwen3.8-27B prefill serves one rank"
        assert not cfg.use_trace, "Qwen3.8-27B prefill runs eager (PREFILL_USE_TRACE=0)"
        self.model = TtModel(self.mesh_device, self.hf_dir, self.layers, cfg.chunk_size, cfg.max_seq_len, lm_head=False)
        assert kv_caches.slots == cfg.num_users, (kv_caches.slots, cfg.num_users)
        self.state = self.model.new_state(cfg.max_seq_len, caches=kv_caches)

    def set_layer_completion_sink(self, sink) -> None:
        """Host acks: ``sink.layers_completed(layer_start, layer_end, slot, pos_start, pos_end)`` (the runner's ring
        writer), or a plain ``sink(global_layer, request_id)``."""
        self._sink = sink

    def set_d2h_ack_service(self, service) -> None:
        self._d2h_service = service

    # ---- serve
    def prefill_chunk(
        self,
        input_tensor,
        kv_caches: K.Qwen38KvCaches,
        *,
        slot_id: int,
        actual_start: int,
        actual_end: int,
        request_id: int = 0,
        d2h_service=None,
        metadata_msg=None,
    ):
        """One chunk [actual_start, actual_end) of slot ``slot_id`` (rows at or past actual_end are the engine's pad).
        Returns None: the single rank is headless, the cache is the output."""
        cfg, model = self.config, self.model
        assert model is not None, "compile() before prefill_chunk()"
        assert kv_caches is self.state.caches, "the runtime serves the cache it was compiled on"
        if not 0 <= slot_id < kv_caches.slots:
            raise ValueError(f"slot {slot_id} outside [0, {kv_caches.slots})")
        if actual_start % cfg.chunk_size:
            raise ValueError(f"chunk start {actual_start} is not a multiple of {cfg.chunk_size} (prefix reuse is off)")
        if not actual_start < actual_end <= min(actual_start + cfg.chunk_size, cfg.max_seq_len):
            raise ValueError(f"chunk [{actual_start}, {actual_end}) outside one chunk of the {cfg.max_seq_len} context")
        d2h = d2h_service if d2h_service is not None else self._d2h_service
        if d2h is not None and metadata_msg is None:
            raise ValueError("D2H acks need the chunk's metadata record")

        self.state.slot = slot_id
        h = model.embed_engine_input(input_tensor)
        if input_tensor.is_allocated():
            ttnn.deallocate(input_tensor)
        for i in self.layers:
            h2 = model.layer(i, h, actual_start, self.state, end=actual_end)
            model.free(h)
            h = h2
            self._ack(i, d2h, metadata_msg, slot_id, actual_start, actual_end, request_id)
        model.free(h)
        if metadata_msg is not None and metadata_msg.is_allocated():
            ttnn.deallocate(metadata_msg)
        return None

    def _ack(self, layer, d2h, metadata_msg, slot_id, start, end, request_id) -> None:
        if d2h is not None:
            ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(d2h, metadata=metadata_msg)
            return
        if self._sink is None:
            return
        ttnn.synchronize_device(self.mesh_device)  # the layer's KV / state is in DRAM before the host hears of it
        if hasattr(self._sink, "layers_completed"):
            self._sink.layers_completed(layer, layer + 1, slot_id, start, end)
        else:
            self._sink(layer, request_id)

    # ---- migration table
    def kv_migration_base_address(self, kv_caches: K.Qwen38KvCaches) -> int:
        """The local address of table entry (table layer 0, pos 0, slot 0, config 0): the K cache's base (shard 0, bank
        0) when the first served layer is a full-attention layer; 0 (the table leaves the entry unset) when it is a
        linear layer, which has no KV records."""
        if kv_caches.kv is None or not kv_caches.attn_layers or kv_caches.attn_layers[0] != self.config.first_layer_idx:
            return 0
        return int(kv_caches.kv.k.buffer_address())

    def kv_migration_stages(self, kv_caches: K.Qwen38KvCaches, first_layer_idx: int, num_my_layers: int):
        from models.demos.common.prefill.runners.migration import KvCacheStage

        if kv_caches.kv is None:
            return []
        return [
            KvCacheStage(int(t.buffer_address()), first_layer_idx, num_my_layers)
            for t in (kv_caches.kv.k, kv_caches.kv.v)
        ]

    def build_kv_chunk_table(
        self, kv_caches: K.Qwen38KvCaches, path, *, first_layer_idx=None, num_my_layers=None, stage_layouts=None
    ):
        first = self.config.first_layer_idx if first_layer_idx is None else first_layer_idx
        n = self.config.num_layers if num_my_layers is None else num_my_layers
        if stage_layouts is not None:
            stages = self.kv_migration_stages(kv_caches, first, n)
            if len(stage_layouts) != len(stages) or any(
                len(g) != 1 or g[0]["base_addr"] != s.base_addr for g, s in zip(stage_layouts, stages)
            ):
                raise ValueError("Qwen3.8-27B migration serves one rank: stage layouts do not match the local cache")
        table = K.build_table(
            self.mesh_device, kv_caches, first_layer_idx=first, num_layers=n, max_seq=self.config.max_seq_len
        )
        ttnn.experimental.disaggregation.export_to_protobuf_file(table, str(path))
        return str(path)
