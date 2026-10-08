# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One fixed traced 4x1K batch with per-request slots, positions and final padding."""

import torch

import ttnn
from models.demos.gemma4_d_p.tt.chunked_batch import ChunkedAttentionLayout, ChunkedBatchPlan, ChunkedRequest
from models.demos.gemma4_d_p.tt.prefill_metadata import PrefillMetadata


class ChunkedBatchRuntime:
    """Attach before populating caches; owns one trace and its stable input buffers.

    Every call has four active requests, each contributing up to 1K tokens.
    Short chunks are final. Output tensors are valid until the next execution;
    ``to_torch`` returns independent host copies. Replaying the staged batch is
    idempotent and is exposed separately for steady-state timing.
    """

    def __init__(self, model, *, num_slots):
        self.model = model
        self.mesh = model.mesh_config
        self.device = model.mesh_device
        self.plan = ChunkedBatchPlan(cp=self.mesh.cp_degree, tp=self.mesh.tp_degree)
        if model.prefill_chunk_size != self.plan.chunk_size or num_slots < self.plan.batch_size:
            raise ValueError("Fixed 4x1K batching requires a 1K model and at least four KV slots")
        for layer in model.layers:
            attention = layer.self_attn
            cache = attention.ring_kv_cache
            tensor = cache.kv if hasattr(cache, "kv") else cache.k
            if tensor.shape[0] // attention.ring_num_layers < num_slots:
                raise ValueError("The supplied model has too few KV slots")
        self.num_slots = num_slots
        self.slot_ends = {}
        self.slot_owners = {}
        self.trace_id = None
        self.output = None
        self.requests = None
        self.completed_requests = None
        self.layer_completion_sink = None
        self.metadata = tuple(PrefillMetadata(self.mesh, clamp_valid=True) for _ in range(self.plan.batch_size))
        self.layout = ChunkedAttentionLayout(self.plan, self.metadata)
        self.input_tokens = ttnn.to_device(self._host([0] * self.plan.packed_size), self.device)
        self.positions = ttnn.to_device(self._host([0] * self.plan.packed_size), self.device)
        self.model.set_prefill_rope_positions(self.positions)
        self.model._prefill_metadata_external = True

    def _host(self, values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32).reshape(1, -1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.device, self.mesh.mesh_shape, dims=(1, None)),
        )

    def validate(self, requests):
        self.plan.validate(
            requests, num_slots=self.num_slots, max_seq_len=self.model.max_seq_len, vocab_size=self.model.vocab_size
        )
        for req in requests:
            if req.actual_start and (
                self.slot_ends.get(req.slot_id) != req.actual_start
                or self.slot_owners.get(req.slot_id) != req.request_id
            ):
                raise ValueError("A continuation must match the slot's request and preceding full chunk")

    def stage(self, requests):
        requests = tuple(
            ChunkedRequest(req.request_id, req.slot_id, req.actual_start, tuple(req.token_ids)) for req in requests
        )
        self.validate(requests)
        # Validate the entire batch before changing any device state.
        ttnn.copy_host_to_device_tensor(self._host(self.plan.pack(requests)), self.input_tokens)
        ttnn.copy_host_to_device_tensor(self._host(self.plan.pack(requests, positions=True)), self.positions)
        for metadata, req in zip(self.metadata, requests):
            metadata.update(slot_idx=req.slot_id, kv_actual_global=req.actual_start, valid_global=req.actual_end)
        self.requests = requests
        self.completed_requests = None

    def _forward(self):
        embeddings = self.model.transform_and_embed_prefill_inputs_device(self.input_tokens)
        return self.model(embeddings, chunked_batch=self.layout)

    def capture(self):
        if self.trace_id is not None:
            raise RuntimeError("The fixed-batch trace is already captured")
        if self.slot_ends:
            raise RuntimeError("Capture must precede real request processing")
        self.stage(tuple(ChunkedRequest(i, i, 0, (0,) * self.plan.chunk_size) for i in range(self.plan.batch_size)))
        warmup = self._forward()
        ttnn.synchronize_device(self.device)
        warmup.deallocate(True)
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.output = self._forward()
        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)
        self.requests = None

    def execute(self):
        """Replay the staged batch; repeat execution rewrites exactly the same cache rows."""
        if self.trace_id is None or self.requests is None:
            raise RuntimeError("Capture and stage a batch before executing")
        ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(self.device)
        self.completed_requests = self.requests
        for req in self.requests:
            self.slot_ends[req.slot_id] = req.actual_end
            self.slot_owners[req.slot_id] = req.request_id
        if self.layer_completion_sink is not None:
            for req in self.requests:
                for layer in range(len(self.model.layers)):
                    self.layer_completion_sink(layer, req.request_id)

    def prefill_batch(self, requests):
        self.stage(requests)
        self.execute()

    def to_torch(self):
        if self.output is None or self.completed_requests is None:
            raise RuntimeError("No completed batch output is available")
        host = ttnn.from_device(self.output, blocking=True)
        shards = ttnn.get_device_tensors(host)
        packed = torch.cat([ttnn.to_torch(shards[rank * self.plan.tp]) for rank in range(self.plan.cp)], dim=-2)
        unpacked = self.plan.unpack(packed)
        return {
            req.request_id: unpacked[lane, : len(req.token_ids)].clone()
            for lane, req in enumerate(self.completed_requests)
        }

    def close(self):
        if self.trace_id is not None:
            ttnn.release_trace(self.device, self.trace_id)
            self.trace_id = None
        if self.output is not None:
            self.output.deallocate(True)
            self.output = None
        self.model.set_prefill_rope_positions(None)
        self.model._prefill_metadata_external = False
        for tensor in (self.input_tokens, self.positions):
            tensor.deallocate(True)
        for metadata in self.metadata:
            metadata.deallocate()
