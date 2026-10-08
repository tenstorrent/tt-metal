# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One active trace variant and explicit output leases for packed prefill batches."""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.gemma4_d_p.tt.prefill_metadata import PrefillMetadata
from models.demos.gemma4_d_p.tt.ragged_prefill import PrefillRequest, RaggedAttentionLayout, RaggedPrefillPlan


@dataclass
class RaggedPrefillResult:
    plan: RaggedPrefillPlan
    requests: tuple[PrefillRequest, ...]
    hidden_states: ttnn.Tensor
    runtime: object
    generation: int
    owns_tensor: bool = False

    def to_torch(self):
        """Copy valid outputs before the next runtime call; returned host tensors are owned."""
        if self.hidden_states is None or self.generation != self.runtime.output_generation:
            raise RuntimeError("This batch output expired; copy outputs before the next runtime call")
        shards = ttnn.get_device_tensors(self.hidden_states)
        packed = torch.cat([ttnn.to_torch(shards[rank * self.plan.tp]) for rank in range(self.plan.cp)], dim=-2)
        return self.plan.unpack(packed, self.requests)

    def deallocate(self):
        if self.owns_tensor and self.hidden_states is not None:
            self.hidden_states.deallocate(True)
        self.hidden_states = None


class RaggedTraceVariant:
    def __init__(self, runtime, plan):
        self.runtime = runtime
        self.plan = plan
        self.trace_id = None
        self.output = None
        self.metadata = tuple(PrefillMetadata(runtime.mesh_config, clamp_valid=True) for _ in plan.segment_sizes)
        self.layout = RaggedAttentionLayout(plan, runtime.mesh_config, self.metadata)
        self.tokens = ttnn.to_device(self._host([0] * plan.packed_size), runtime.mesh_device)
        self.positions = ttnn.to_device(self._host([0] * plan.packed_size), runtime.mesh_device)

    def _host(self, values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int64).reshape(1, self.plan.packed_size),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.runtime.mesh_device, self.runtime.config.mesh_shape, dims=(1, None)
            ),
        )

    def stage(self, requests):
        for tensor, values in (
            (self.tokens, self.plan.pack(requests)),
            (self.positions, self.plan.pack(requests, positions=True)),
        ):
            ttnn.copy_host_to_device_tensor(self._host(values), tensor)
        for metadata, request in zip(self.metadata, requests):
            metadata.update(
                slot_idx=request.slot_id, kv_actual_global=request.actual_start, valid_global=request.actual_end
            )

    def forward(self):
        model = self.runtime.model
        return model(
            model.transform_and_embed_prefill_inputs_device(self.tokens),
            ragged_layout=self.layout,
            rope_positions=self.positions,
        )

    def run(self, *, use_trace):
        device = self.runtime.mesh_device
        if not use_trace:
            return self.forward()
        if self.trace_id is None:
            # Warmup and capture rewrite the same current chunks. No completion
            # acknowledgements are emitted until the actual replay has finished.
            warmup = self.forward()
            ttnn.synchronize_device(device)
            warmup.deallocate(True)
            self.trace_id = ttnn.begin_trace_capture(device, cq_id=0)
            try:
                self.output = self.forward()
            finally:
                ttnn.end_trace_capture(device, self.trace_id, cq_id=0)
            ttnn.synchronize_device(device)
        ttnn.execute_trace(device, self.trace_id, cq_id=0, blocking=False)
        # Borrow the captured output. Allocating a retained clone after capture
        # would let the next replay overwrite it through transient-buffer reuse.
        return self.output

    def release(self):
        if self.trace_id is not None:
            ttnn.release_trace(self.runtime.mesh_device, self.trace_id)
            self.trace_id = None
        if self.output is not None:
            self.output.deallocate(True)
            self.output = None
        self.tokens.deallocate(True)
        self.positions.deallocate(True)
        for metadata in self.metadata:
            metadata.deallocate()
