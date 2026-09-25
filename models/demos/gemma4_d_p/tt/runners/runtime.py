# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One traced prefill chunk at a time into engine-owned Gemma4 KV slots."""

import torch

import ttnn
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.tt.common import create_tt_model


class Gemma4PrefillRuntime:
    def __init__(self, *, mesh_device, hf_model_id, tt_cache_path, config):
        self.mesh_device = mesh_device
        self.mesh_config = MeshConfig(mesh_device)
        self.hf_model_id = hf_model_id
        self.tt_cache_path = tt_cache_path
        self.config = config
        self.trace_id = None
        self.d2h_service = None
        self.layer_completion_sink = None
        self.slot_ends = [0] * config.num_users

    def _host_tokens(self, values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int64).reshape(1, self.config.chunk_size),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, self.config.mesh_shape, dims=(1, None)),
        )

    def make_chunk_input(self, token_ids):
        return ttnn.to_device(self._host_tokens(token_ids), self.mesh_device)

    def _stage_positions(self, slot_id, actual_start):
        self.model.prefill_metadata.update(slot_idx=slot_id, kv_actual_global=actual_start)
        positions = range(actual_start, actual_start + self.config.chunk_size)
        ttnn.copy_host_to_device_tensor(self._host_tokens(positions), self.positions)

    def _forward(self):
        embeddings = self.model.transform_and_embed_prefill_inputs_device(self.input_tokens)
        return self.model(
            hidden_states=embeddings,
            d2h_service=self.d2h_service,
            metadata_msg=self.metadata,
        )

    def compile(self, kv_cache):
        _, self.model, _, _ = create_tt_model(
            mesh_config=self.mesh_config,
            hf_model_id=self.hf_model_id,
            prefill_chunk_size=self.config.chunk_size,
            max_seq_len=self.config.max_seq_len,
            max_batch_size=1,
            ring_kv_caches=kv_cache,
            tt_cache_path=self.tt_cache_path,
        )
        self.input_tokens = self.make_chunk_input([0] * self.config.chunk_size)
        self.positions = self.make_chunk_input(range(self.config.chunk_size))
        self.metadata = ttnn.from_torch(
            torch.tensor([0, 0, self.config.chunk_size], dtype=torch.int64).reshape(1, 1, 1, 3),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        self.model.set_prefill_rope_positions(self.positions)
        self.model._prefill_metadata_external = True
        self._stage_positions(0, 0)
        output = self._forward()
        ttnn.synchronize_device(self.mesh_device)
        output.deallocate(True)

    def set_d2h_ack_service(self, service):
        self.d2h_service = service

    def set_layer_completion_sink(self, sink):
        self.layer_completion_sink = sink

    def warmup_ack_count(self):
        return self.config.num_layers if self.d2h_service is not None else 0

    def capture_trace(self, kv_cache):
        self._check_cache(kv_cache)
        if self.d2h_service is not None:
            output = self._forward()
            ttnn.synchronize_device(self.mesh_device)
            output.deallocate(True)
        self.trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self.output = self._forward()
        ttnn.end_trace_capture(self.mesh_device, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)

    def _check_cache(self, kv_cache):
        if len(kv_cache.layers) != len(self.model.layers) or any(
            layer.self_attn.ring_kv_cache is not cache for layer, cache in zip(self.model.layers, kv_cache.layers)
        ):
            raise ValueError("The traced runtime requires the caches supplied to compile")

    def validate_chunk(self, slot_id, actual_start, actual_end):
        if not 0 <= slot_id < self.config.num_users:
            raise ValueError(f"KV slot {slot_id} is outside [0, {self.config.num_users})")
        if actual_start < 0 or actual_start % self.config.chunk_size:
            raise ValueError("Chunk start must be a nonnegative multiple of 8192")
        if not actual_start < actual_end <= min(actual_start + self.config.chunk_size, self.config.max_seq_len):
            raise ValueError("Chunk must contain 1 to 8192 real tokens within the 256K context")
        if actual_start != 0 and actual_start != self.slot_ends[slot_id]:
            raise ValueError(f"Slot {slot_id} expects position {self.slot_ends[slot_id]}, got {actual_start}")

    def prefill_chunk(
        self,
        input_tensor,
        kv_cache,
        *,
        slot_id,
        actual_start,
        actual_end,
        request_id=0,
        d2h_service=None,
        metadata_msg=None,
    ):
        self.validate_chunk(slot_id, actual_start, actual_end)
        self._check_cache(kv_cache)
        if self.trace_id is None:
            raise RuntimeError("capture_trace must run before serving chunks")
        if d2h_service is not self.d2h_service:
            raise ValueError("The D2H service must match the captured service")
        tokens = ttnn.reshape(input_tensor, (1, self.config.chunk_size // self.mesh_config.cp_degree))
        ttnn.copy(tokens, self.input_tokens)
        if metadata_msg is not None:
            ttnn.copy(metadata_msg, self.metadata)
        elif self.d2h_service is not None:
            raise ValueError("D2H acknowledgments require request metadata")
        self._stage_positions(slot_id, actual_start)
        ttnn.execute_trace(self.mesh_device, self.trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.mesh_device)
        self.slot_ends[slot_id] = actual_end
        if self.layer_completion_sink is not None:
            for layer_idx in range(self.config.num_layers):
                self.layer_completion_sink(layer_idx, request_id)
        ttnn.deallocate(input_tensor)
        if metadata_msg is not None:
            ttnn.deallocate(metadata_msg)

    def build_kv_chunk_table(self, kv_cache, path):
        from models.demos.gemma4_d_p.tt.runners.kv_chunk_table import build_and_serialize_kv_chunk_table

        self._check_cache(kv_cache)
        return build_and_serialize_kv_chunk_table(
            path=path,
            mesh_device=self.mesh_device,
            kv_caches=kv_cache,
            chunk_size=self.config.chunk_size,
        )

    def release_trace(self):
        if self.trace_id is not None:
            ttnn.release_trace(self.mesh_device, self.trace_id)
            self.trace_id = None
        self.d2h_service = None
        self.layer_completion_sink = None
