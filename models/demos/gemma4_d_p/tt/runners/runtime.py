# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Packed striped prefill with a bounded cache of traces for request layouts."""

import os
from collections import OrderedDict

import torch

import ttnn
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.tt.common import create_tt_model
from models.demos.gemma4_d_p.tt.runners.packing import (
    PrefillSegment,
    pack_positions,
    pack_tokens,
    stripe_rows,
    validate_segments,
)


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
        self.traces = OrderedDict()
        self.max_traces = int(os.environ.get("GEMMA4_PREFILL_MAX_TRACES", "4"))
        if self.max_traces < 1:
            raise ValueError("GEMMA4_PREFILL_MAX_TRACES must be positive")
        self._next_request_id = 0
        self._ready = False

    def _host_rows(self, rows):
        return ttnn.from_torch(
            torch.tensor(rows, dtype=torch.int64).reshape(1, self.config.chunk_size),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, self.config.mesh_shape, dims=(1, None)),
        )

    def _host_tokens(self, values):
        return self._host_rows(stripe_rows(list(values), self.mesh_config.cp_degree))

    def make_chunk_input(self, token_ids):
        return ttnn.to_device(self._host_tokens(token_ids), self.mesh_device)

    def make_batch_input(self, token_lists, segments):
        self._validate_batch(segments)
        rows = pack_tokens(token_lists, segments, chunk_size=self.config.chunk_size, cp=self.mesh_config.cp_degree)
        return ttnn.to_device(self._host_rows(rows), self.mesh_device)

    def _stage_batch(self, segments):
        self.model.packed_prefill_metadata.update(
            (
                segment.slot_id,
                segment.actual_start,
                segment.actual_end - segment.actual_start,
                segment.padded_length(self.mesh_config.cp_degree),
            )
            for segment in segments
        )
        rows = pack_positions(segments, chunk_size=self.config.chunk_size, cp=self.mesh_config.cp_degree)
        ttnn.copy_host_to_device_tensor(self._host_rows(rows), self.positions)
        self._layout = tuple(s.padded_length(self.mesh_config.cp_degree) for s in segments)

    def _stage_positions(self, slot_id, actual_start):
        self._stage_batch(
            (
                PrefillSegment(
                    slot_id, actual_start, min(actual_start + self.config.chunk_size, self.config.max_seq_len)
                ),
            )
        )

    def _forward(self):
        embeddings = self.model.transform_and_embed_prefill_inputs_device(self.input_tokens)
        result = self.model(hidden_states=embeddings, prefill_lengths=self._layout)
        if getattr(self, "output", None) is None:
            return result
        ttnn.copy(result, self.output)
        result.deallocate(True)

    def compile(self, kv_cache):
        _, self.model, _, _ = create_tt_model(
            mesh_config=self.mesh_config,
            hf_model_id=self.hf_model_id,
            prefill_chunk_size=self.config.chunk_size,
            max_seq_len=self.config.max_seq_len,
            max_batch_size=1,
            ring_kv_caches=kv_cache,
            tt_cache_path=self.tt_cache_path,
            num_layers=self.config.num_layers if self.config.num_layers != 60 else None,
            stripe_size=32,
            max_prefill_requests=min(
                self.config.num_users, self.config.chunk_size // (32 * self.mesh_config.cp_degree)
            ),
        )
        self.input_tokens = self.make_chunk_input([0] * self.config.chunk_size)
        self.positions = self.make_chunk_input([0] * self.config.chunk_size)
        self.model.set_prefill_rope_positions(self.positions)
        self.model._prefill_metadata_external = True
        self._stage_positions(0, 0)
        self.output = self._forward()
        ttnn.synchronize_device(self.mesh_device)

    def set_d2h_ack_service(self, service):
        if service is not None:
            raise ValueError("Packed Gemma4 prefill requires PREFILL_LAYER_ACK_D2H=0")

    def set_layer_completion_sink(self, sink):
        self.layer_completion_sink = sink

    def warmup_ack_count(self):
        return 0

    def capture_trace(self, kv_cache):
        self._check_cache(kv_cache)
        self._ready = True

    def _capture_layout(self):
        while len(self.traces) >= self.max_traces:
            _, trace_id = self.traces.popitem(last=False)
            ttnn.release_trace(self.mesh_device, trace_id)
        self._forward()
        ttnn.synchronize_device(self.mesh_device)
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        try:
            self._forward()
        except Exception:
            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
            ttnn.release_trace(self.mesh_device, trace_id)
            raise
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        self.traces[self._layout] = trace_id

    def _check_cache(self, kv_cache):
        if len(kv_cache.layers) != len(self.model.layers) or any(
            layer.self_attn.ring_kv_cache is not cache for layer, cache in zip(self.model.layers, kv_cache.layers)
        ):
            raise ValueError("The traced runtime requires the caches supplied to compile")

    def _validate_batch(self, segments):
        validate_segments(
            segments,
            chunk_size=self.config.chunk_size,
            cp=self.config.mesh_shape[self.config.sp_axis],
            num_users=self.config.num_users,
            max_seq_len=self.config.max_seq_len,
            slot_ends=self.slot_ends,
        )

    def validate_chunk(self, slot_id, actual_start, actual_end):
        self._validate_batch((PrefillSegment(slot_id, actual_start, actual_end),))

    def prefill_batch(self, input_tensor, kv_cache, *, segments, request_ids=None, d2h_service=None, metadata_msg=None):
        segments = tuple(segments)
        self._validate_batch(segments)
        self._check_cache(kv_cache)
        if not self._ready:
            raise RuntimeError("capture_trace must run before serving batches")
        if d2h_service is not None:
            raise ValueError("Packed prefill uses host completion acknowledgments")
        if request_ids is None:
            request_ids = range(self._next_request_id, self._next_request_id + len(segments))
        request_ids = tuple(request_ids)
        if len(request_ids) != len(segments):
            raise ValueError("Each request needs one completion ID")
        tokens = ttnn.reshape(input_tensor, (1, self.config.chunk_size // self.mesh_config.cp_degree))
        ttnn.copy(tokens, self.input_tokens)
        ttnn.deallocate(input_tensor)
        if metadata_msg is not None:
            ttnn.deallocate(metadata_msg)
        self._stage_batch(segments)
        if self.config.use_trace:
            if self._layout not in self.traces:
                self._capture_layout()
            self.traces.move_to_end(self._layout)
            self.trace_id = self.traces[self._layout]
            ttnn.execute_trace(self.mesh_device, self.trace_id, cq_id=0, blocking=False)
        else:
            self._forward()
        ttnn.synchronize_device(self.mesh_device)
        for segment in segments:
            self.slot_ends[segment.slot_id] = segment.actual_end
            if self.layer_completion_sink is not None:
                self.layer_completion_sink.layers_completed(
                    0, self.config.num_layers, segment.slot_id, segment.actual_start, segment.actual_end
                )
        self._next_request_id += len(segments)

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
        return self.prefill_batch(
            input_tensor,
            kv_cache,
            segments=(PrefillSegment(slot_id, actual_start, actual_end),),
            request_ids=(request_id,),
            d2h_service=d2h_service,
            metadata_msg=metadata_msg,
        )

    def kv_migration_stages(self, kv_cache, first_layer_idx, num_my_layers):
        from models.demos.common.prefill.runners.migration import KvCacheStage

        self._check_cache(kv_cache)
        if first_layer_idx != 0 or num_my_layers != self.config.num_layers:
            raise ValueError("Gemma4 migration requires all layers on one rank")
        stages = []
        for layer_idx, cache in enumerate(kv_cache.layers):
            tensors = (cache.kv,) if hasattr(cache, "kv") else (cache.k, cache.v)
            stages.extend(KvCacheStage(int(tensor.buffer_address()), layer_idx, 1) for tensor in tensors)
        return stages

    def build_kv_chunk_table(self, kv_cache, path, *, first_layer_idx=0, num_my_layers=None, stage_layouts=None):
        from models.demos.gemma4_d_p.tt.runners.kv_chunk_table import build_and_serialize_kv_chunk_table

        self._check_cache(kv_cache)
        stages = self.kv_migration_stages(
            kv_cache, first_layer_idx, self.config.num_layers if num_my_layers is None else num_my_layers
        )
        if stage_layouts is not None:
            if len(stage_layouts) != len(stages):
                raise ValueError("Gemma4 migration requires one stage per cache tensor")
            for gathered, stage in zip(stage_layouts, stages):
                if len(gathered) != 1 or any(
                    gathered[0][key] != value
                    for key, value in (
                        ("rank", 0),
                        ("base_addr", stage.base_addr),
                        ("first_layer", stage.first_layer),
                        ("count", stage.count),
                    )
                ):
                    raise ValueError("Gemma4 migration stage does not match its single-rank cache")
        return build_and_serialize_kv_chunk_table(
            path=path,
            mesh_device=self.mesh_device,
            kv_caches=kv_cache,
            chunk_size=32 * self.mesh_config.cp_degree,
        )

    def release_trace(self):
        for trace_id in self.traces.values():
            ttnn.release_trace(self.mesh_device, trace_id)
        self.traces.clear()
        if getattr(self, "output", None) is not None:
            self.output.deallocate(True)
        self.trace_id = None
        self.output = None
        self._ready = False
        self.layer_completion_sink = None
