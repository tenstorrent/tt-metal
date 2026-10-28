# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MiMo's prefill runtime for tt-metal's prefill runner (models/demos/common/prefill/runners/prefill_runner.py): one
chunk at a time into the engine-owned served caches (kv_caches.py), the device model of tt/model.py on them.

Per chunk: the chunk's metadata (slot, start, end, positions, masks) is staged into the model's device buffers, the
engine's H2D tokens ([sp, 1, chunk / sp] uint32, block-cyclic rows, 0xFFFFFFFF past actual_end) are embedded on the
device, and each served layer runs; after each layer one completion is reported: a D2H record (the chunk's metadata,
enqueued on the same CQ right after the layer, so it follows that layer's KV and zeroed pad rows) when the runner
gave a D2H service, else the host sink after a device sync. Eager (no trace).
"""

from __future__ import annotations

import ttnn
from models.demos.common.prefill.runners.migration import KvCacheStage

from .kv_chunk_table import RECORD, build_and_serialize_kv_chunk_table


class _Spec:
    """The three spec values MimoModel reads, from the runner's params."""

    def __init__(self, hf_dir, chunk: int, seq: int):
        self.hf_dir = hf_dir
        self._v = {"target.chunk": chunk, "target.seq": seq}

    def get(self, key, default=None):
        return self._v.get(key, default)


class MimoPrefillRuntime:
    def __init__(self, *, mesh_device, hf_dir, config):
        self.mesh_device = mesh_device
        self.hf_dir = hf_dir
        self.config = config
        self.model = None
        self.kv = None
        self.d2h_service = None
        self.layer_completion_sink = None
        self.layers = list(range(config.first_layer_idx, config.first_layer_idx + config.num_layers))

    # ---- setup
    def compile(self, kv_caches):
        from models.demos.mimo_v26_flash_rl.tt.model import MimoModel, State

        self._check_cache(kv_caches, compiled=False)
        spec = _Spec(self.hf_dir, self.config.chunk_size, kv_caches.max_seq_len)  # the caches' capacity
        self.model = MimoModel(
            self.mesh_device,
            spec,
            self.layers,
            lm_head=False,
            slots=kv_caches.num_users,
            kv_caches=kv_caches.layers,
        )
        self.state = State(self.model)

    def set_d2h_ack_service(self, service):
        self.d2h_service = service

    def set_layer_completion_sink(self, sink):
        self.layer_completion_sink = sink

    def _check_cache(self, kv_caches, compiled: bool = True):
        """The runtime serves one cache handle: the first one it is given (the table may be built before compile)."""
        if compiled and self.model is None:
            raise RuntimeError("compile must run before serving chunks")
        if self.kv is None:
            if kv_caches.layer_ids != self.layers:
                raise ValueError(f"caches hold layers {kv_caches.layer_ids}, the runtime serves {self.layers}")
            self.kv = kv_caches
        elif kv_caches is not self.kv:
            raise ValueError("the runtime serves the caches it was first given")

    # ---- serving
    def validate_chunk(self, slot_id, actual_start, actual_end):
        cfg = self.config
        if not 0 <= slot_id < cfg.num_users:
            raise ValueError(f"KV slot {slot_id} is outside [0, {cfg.num_users})")
        if actual_start < 0 or actual_start % RECORD:
            raise ValueError(f"chunk start {actual_start} is not a nonnegative multiple of {RECORD}")
        if actual_start + cfg.chunk_size > cfg.max_seq_len:
            raise ValueError(
                f"chunk [{actual_start}, {actual_start + cfg.chunk_size}) runs past the {cfg.max_seq_len}-token slot"
            )
        if not actual_start < actual_end <= actual_start + cfg.chunk_size:
            raise ValueError(f"chunk must hold 1 to {cfg.chunk_size} real tokens, got [{actual_start}, {actual_end})")

    def _ack(self, layer, slot_id, actual_start, actual_end, request_id, d2h_service, metadata_msg):
        if d2h_service is not None:
            # same CQ, after the layer's cache write and pad zero: the record cannot reach the host before them
            ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(d2h_service, metadata=metadata_msg)
        sink = self.layer_completion_sink
        if sink is not None:
            ttnn.synchronize_device(self.mesh_device)
            if hasattr(sink, "layers_completed"):  # the runner's ring writer: global layer ids
                sink.layers_completed(layer, layer + 1, slot_id, actual_start, actual_end)
            else:  # a plain callable (layer in the table, request id)
                sink(layer - self.config.first_layer_idx, request_id)

    def prefill_chunk(
        self,
        input_tensor,
        kv_caches,
        *,
        slot_id,
        actual_start,
        actual_end,
        request_id=0,
        d2h_service=None,
        metadata_msg=None,
        **kwargs,
    ):
        self._check_cache(kv_caches)
        self.validate_chunk(slot_id, actual_start, actual_end)
        d2h_service = d2h_service if d2h_service is not None else self.d2h_service
        if d2h_service is not None and metadata_msg is None:
            raise ValueError("D2H layer acknowledgements need the chunk's metadata record")
        m = self.model
        m._stage(actual_start, actual_end, slot_id)  # the runtime's metadata step (host -> device buffers)
        h = m.embed_device(input_tensor)  # consumes input_tensor
        for layer in self.layers:
            out = m.layer(layer, h, actual_start, self.state, end=actual_end)
            m.free(h)
            h = out
            self._ack(layer, slot_id, actual_start, actual_end, request_id, d2h_service, metadata_msg)
        m.free(h)
        ttnn.synchronize_device(self.mesh_device)
        if metadata_msg is not None:
            ttnn.deallocate(metadata_msg)
        return None

    # ---- migration
    def kv_migration_base_address(self, kv_caches) -> int:
        """Table entry 0: layer first_layer_idx's key, slot 0, head 0, position 0 (bank 0, the buffer's base)."""
        self._check_cache(kv_caches, compiled=False)
        return int(kv_caches[self.layers[0]].key.buffer_address())

    def kv_migration_stages(self, kv_caches, first_layer_idx, num_my_layers):
        self._check_cache(kv_caches, compiled=False)
        if (first_layer_idx, num_my_layers) != (self.config.first_layer_idx, self.config.num_layers):
            raise ValueError(
                f"migration of layers [{first_layer_idx}, +{num_my_layers}) != the runtime's {self.layers}"
            )
        stages = []
        for layer in self.layers:
            for t in kv_caches[layer].tensors().values():
                stages.append(KvCacheStage(int(t.buffer_address()), layer, 1))
        return stages

    def build_kv_chunk_table(self, kv_caches, path, *, first_layer_idx=None, num_my_layers=None, stage_layouts=None):
        self._check_cache(kv_caches, compiled=False)
        first = self.config.first_layer_idx if first_layer_idx is None else first_layer_idx
        n = self.config.num_layers if num_my_layers is None else num_my_layers
        stages = self.kv_migration_stages(kv_caches, first, n)
        if stage_layouts is not None:
            if len(stage_layouts) != len(stages):
                raise ValueError("MiMo migration needs one gathered layout per cache stage")
            for gathered, stage in zip(stage_layouts, stages):
                if len(gathered) != 1 or (
                    gathered[0]["base_addr"],
                    gathered[0]["first_layer"],
                    gathered[0]["count"],
                ) != (
                    stage.base_addr,
                    stage.first_layer,
                    stage.count,
                ):
                    raise ValueError("MiMo migration stage does not match its single-rank cache")
        return build_and_serialize_kv_chunk_table(
            path=path, mesh_device=self.mesh_device, kv=kv_caches, first_layer_idx=first
        )
