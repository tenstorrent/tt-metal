# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batched prefill steps: several requests' chunks stacked in one traced step (continuous-batching PoC).

Shared by tests/test_batched_prefill.py and demo/batching_server.py. StepRunner.step times the trace replay only
(execute + sync, the device step time); host staging comes before it.
"""

import os
import statistics
import time

import torch
from loguru import logger

import ttnn
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.demo.text_demo_prefill import _cp_or_replicate_mapper, _hf_model_id
from models.demos.gemma4_d_p.tt.common import create_tt_model
from models.demos.gemma4_d_p.tt.prefill_metadata import PrefillLanes


def build_batched_model(mesh_device, chunk_size, num_slots, capacity, hf_model_id=None):
    """A Gemma4 prefill model with num_slots KV slots of `capacity` tokens and every lane's metadata allocated."""
    mesh_config = MeshConfig(mesh_device)
    model_args, model, _, _ = create_tt_model(
        mesh_config=mesh_config,
        prefill_chunk_size=chunk_size,
        max_batch_size=num_slots,
        max_seq_len=capacity,
        dtype=ttnn.bfloat16,
        hf_model_id=hf_model_id or _hf_model_id(),
    )
    # Every lane's metadata up front, before any activation is live (see lane_prefill_metadata).
    model.lane_prefill_metadata(num_slots)
    return mesh_config, model_args, model


class StepRunner:
    """Stacked inputs, per-lane metadata and one trace per layout (each lane's chunk width, in lane order) for a model
    with several KV slots. A lane is (slot, start, tokens); its width is len(tokens)."""

    def __init__(self, mesh_device, mesh_config, model, chunk_size):
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.model = model
        self.chunk = chunk_size
        self.inputs = {}
        self.traces = {}
        self.outputs = {}
        # Mixed widths: every step, even one plain chunk, takes the PrefillLanes path (residual spilled across
        # attention). The first compile's all-gathers create their global semaphores lazily wherever L1 is free; with
        # the residual in L1 they land at ~1,440,576, under a 4k-wide global SDPA's CB region (ends at 1,475,712).
        self.always_lanes = False
        self.reps = int(os.environ.get("G4B_REPS", "5"))

    @staticmethod
    def layout(lanes):
        return tuple(len(tokens) for _, _, tokens in lanes)

    def _host(self, seqs):
        """Per-lane 1D int sequences -> the [1, total] host tensor whose CP shard is each rank's stacked slabs."""
        cp = self.mesh_config.cp_degree
        parts = []
        for rank in range(cp):
            for seq in seqs:
                slab = len(seq) // cp
                parts.append(seq[rank * slab : (rank + 1) * slab])
        return ttnn.from_torch(
            torch.cat(parts).reshape(1, -1).to(torch.int32).contiguous(),
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=_cp_or_replicate_mapper(self.mesh_config, seq_dim=-1),
        )

    def _ensure(self, layout):
        total = sum(layout)
        if total not in self.inputs:
            zeros = [torch.zeros(total, dtype=torch.int32)]
            self.inputs[total] = tuple(ttnn.to_device(self._host(zeros), device=self.mesh_device) for _ in range(2))
        self.model.lane_prefill_metadata(len(layout))
        if len(layout) > 1:
            self.model.lane_vector_metadata(len(layout))

    def stage(self, lanes):
        """Host refresh of everything a replay reads."""
        tokens, positions = self.inputs[sum(self.layout(lanes))]
        ttnn.copy_host_to_device_tensor(self._host([t for _, _, t in lanes]), tokens)
        starts = [torch.arange(start, start + len(t)) for _, start, t in lanes]
        ttnn.copy_host_to_device_tensor(self._host(starts), positions)
        for metadata, (slot, start, _) in zip(self.model.lane_prefill_metadata(len(lanes)), lanes):
            metadata.update(slot_idx=slot, kv_actual_global=start)
        if len(lanes) > 1:
            self.model.lane_vector_metadata(len(lanes)).update(
                slot_idx=[slot for slot, _, _ in lanes], kv_actual_global=[start for _, start, _ in lanes]
            )

    def _forward(self, layout):
        tokens, positions = self.inputs[sum(layout)]
        self.model.set_prefill_rope_positions(positions)
        embeds = self.model.transform_and_embed_prefill_inputs_device(tokens)
        metadata = self.model.lane_prefill_metadata(len(layout))
        rows = [w // self.mesh_config.cp_degree for w in layout]
        if layout == (self.chunk,) and not self.always_lanes:
            metadata = metadata[0]
        else:
            # Each request's rows (one request wider than the model's chunk, or always_lanes, too), and for several
            # requests the B-element slot / prefix tensors the lanes ring SDPA reads.
            vector = self.model.lane_vector_metadata(len(layout)) if len(layout) > 1 else None
            metadata = PrefillLanes(metadata, rows, vector=vector)
        return self.model(hidden_states=embeds, prefill_metadata=metadata)

    def capture_all(self, lane_sets):
        """Compile every layout first, then capture every trace.

        A compile pass allocates persistent tensors lazily (gather-index constants, per-lane ring buffers). Allocated
        after another trace's capture, they can land on that trace's freed intermediates, and each replay of it then
        overwrites them. So no compile may follow a capture.
        """
        for lanes in lane_sets:
            layout = self.layout(lanes)
            self._ensure(layout)
            t0 = time.time()
            self.stage(lanes)
            out = self._forward(layout)
            ttnn.synchronize_device(self.mesh_device)
            out.deallocate(True)
            logger.info(f"[batching] layout={layout} compile={time.time() - t0:.1f}s")
        for lanes in lane_sets:
            layout = self.layout(lanes)
            self.stage(lanes)
            tid = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            self.outputs[layout] = self._forward(layout)
            ttnn.end_trace_capture(self.mesh_device, tid, cq_id=0)
            ttnn.synchronize_device(self.mesh_device)
            self.traces[layout] = tid
            logger.info(f"[batching] layout={layout} captured")

    def step(self, lanes):
        """Stage and replay one step; returns its execute+sync time in ms."""
        self.stage(lanes)
        t0 = time.time()
        ttnn.execute_trace(self.mesh_device, self.traces[self.layout(lanes)], cq_id=0, blocking=False)
        ttnn.synchronize_device(self.mesh_device)
        return (time.time() - t0) * 1000

    def timed(self, lanes):
        """Median replay time in ms over G4B_REPS replays, after one warm-up."""
        self.step(lanes)
        return statistics.median(self.step(lanes) for _ in range(self.reps))

    def release(self):
        for tid in self.traces.values():
            ttnn.release_trace(self.mesh_device, tid)
        self.traces = {}
