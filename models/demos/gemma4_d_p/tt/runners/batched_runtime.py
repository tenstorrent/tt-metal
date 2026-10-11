# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batched traced prefill steps into engine-owned Gemma4 KV slots: up to max_lanes chunks of 2048 tokens per step.

Serves the multi-lane step message (models/demos/common/prefill/step_message.py): each lane is one chunk of one
request, a request may take several adjacent lanes, and every chunk is chunk_size wide, so every request's KV has
the one layout the migration table is built for, however its chunks were grouped into steps. One trace per lane
count, all compiled before any is captured (see BATCHING_POC.md, issue 2). Projections run at HiFi2, as an
unbatched 2k chunk does: stacked rows would otherwise switch them to LoFi, which fails the KV gates.

Layer acks: lane 0's ack per layer as the unbatched runtime emits it, then num_layers more per further lane after
the last layer, so a step of n lanes acks like n chunks in lane order and no ack precedes its KV.
"""

import torch

import ttnn
from models.demos.common.prefill import step_message
from models.demos.gemma4_d_p.tt.attention import operations as attention_operations
from models.demos.gemma4_d_p.tt.common import create_tt_model
from models.demos.gemma4_d_p.tt.prefill_metadata import PrefillLanes
from models.demos.gemma4_d_p.tt.runners.runtime import Gemma4PrefillRuntime


class Gemma4BatchedPrefillRuntime(Gemma4PrefillRuntime):
    def __init__(self, *, mesh_device, hf_model_id, tt_cache_path, config):
        super().__init__(mesh_device=mesh_device, hf_model_id=hf_model_id, tt_cache_path=tt_cache_path, config=config)
        self.max_lanes = config.max_lanes
        self.lane_counts = range(self.max_lanes, 0, -1)  # widest first
        self.slab = config.chunk_size // self.mesh_config.cp_degree
        self.inputs = {}
        self.positions = {}
        self.traces = {}
        self.outputs = {}
        self.saved_fidelity_override = None

    def _host_rows(self, seqs):
        """Per-lane chunk-long sequences -> the [1, total] host tensor whose CP shard is each rank's stacked slabs,
        the layout the step message's payload carries (step_message.stacked_rows)."""
        cp, n = self.mesh_config.cp_degree, len(seqs)
        rows = step_message.stacked_rows(seqs, self.config.chunk_size, cp, n)
        return ttnn.from_torch(
            rows.reshape(1, -1).to(torch.int32).contiguous(),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, self.config.mesh_shape, dims=(1, None)),
        )

    def _zero_lanes(self, n):
        return [step_message.Lane(0, k * self.config.chunk_size, (k + 1) * self.config.chunk_size) for k in range(n)]

    def _stage(self, lanes):
        """Host refresh of the positions and per-lane metadata a replay reads."""
        n = len(lanes)
        starts = [torch.arange(lane.start, lane.start + self.config.chunk_size) for lane in lanes]
        ttnn.copy_host_to_device_tensor(self._host_rows(starts), self.positions[n])
        for metadata, lane in zip(self.model.lane_prefill_metadata(n), lanes):
            metadata.update(slot_idx=lane.slot, kv_actual_global=lane.start)
        if n > 1:
            self.model.lane_vector_metadata(n).update(
                slot_idx=[lane.slot for lane in lanes], kv_actual_global=[lane.start for lane in lanes]
            )

    def _forward_lanes(self, n):
        self.model.set_prefill_rope_positions(self.positions[n])
        embeddings = self.model.transform_and_embed_prefill_inputs_device(self.inputs[n])
        metadata = self.model.lane_prefill_metadata(n)
        if n == 1:
            metadata = metadata[0]
        else:
            metadata = PrefillLanes(metadata, [self.slab] * n, vector=self.model.lane_vector_metadata(n))
        output = self.model(
            hidden_states=embeddings,
            prefill_metadata=metadata,
            d2h_service=self.d2h_service,
            metadata_msg=self.metadata,
        )
        if self.d2h_service is not None:
            for _ in range((n - 1) * self.config.num_layers):
                ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(
                    self.d2h_service, metadata=self.metadata
                )
        return output

    def _run_once(self, n):
        """An untraced n-lane step on slot 0 (compiles its programs)."""
        self._stage(self._zero_lanes(n))
        output = self._forward_lanes(n)
        ttnn.synchronize_device(self.mesh_device)
        output.deallocate(True)

    def compile(self, kv_cache):
        # Until release_trace. A finer fix would choose fidelity by each lane's rows rather than the stacked rows.
        self.saved_fidelity_override = attention_operations.PROJECTION_FIDELITY_OVERRIDE
        attention_operations.PROJECTION_FIDELITY_OVERRIDE = ttnn.MathFidelity.HiFi2
        _, self.model, _, _ = create_tt_model(
            mesh_config=self.mesh_config,
            hf_model_id=self.hf_model_id,
            prefill_chunk_size=self.config.chunk_size,
            max_seq_len=self.config.max_seq_len,
            max_batch_size=self.config.num_users,
            ring_kv_caches=kv_cache,
            tt_cache_path=self.tt_cache_path,
        )
        # Every lane's metadata, inputs and positions before any activation is live.
        self.model.lane_prefill_metadata(self.max_lanes)
        for n in self.lane_counts:
            if n > 1:
                self.model.lane_vector_metadata(n)
            zeros = [torch.zeros(self.config.chunk_size, dtype=torch.int32)] * n
            self.inputs[n] = ttnn.to_device(self._host_rows(zeros), self.mesh_device)
            self.positions[n] = ttnn.to_device(self._host_rows(zeros), self.mesh_device)
        words = step_message.metadata_words(self.max_lanes)
        self.metadata = ttnn.from_torch(
            torch.zeros(1, 1, 1, words, dtype=torch.int64),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        for n in self.lane_counts:
            self._run_once(n)

    def capture_trace(self, kv_cache):
        self._check_cache(kv_cache)
        if self.d2h_service is not None:
            # Compiles the ack op (warmup_ack_count records); every other program is compiled already.
            self._run_once(1)
        for n in self.lane_counts:
            self._stage(self._zero_lanes(n))
            trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            self.outputs[n] = self._forward_lanes(n)
            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(self.mesh_device)
            self.traces[n] = trace_id

    def validate_step(self, lanes):
        step_message.validate_lanes(
            lanes,
            chunk_size=self.config.chunk_size,
            max_lanes=self.max_lanes,
            max_seq_len=self.config.max_seq_len,
            num_slots=self.config.num_users,
        )
        for i, lane in enumerate(lanes):
            first_of_slot = i == 0 or lanes[i - 1].slot != lane.slot
            if first_of_slot and lane.start != 0 and lane.start != self.slot_ends[lane.slot]:
                raise ValueError(f"Slot {lane.slot} expects position {self.slot_ends[lane.slot]}, got {lane.start}")

    def prefill_step(self, input_tensor, kv_cache, *, lanes, request_id=0, d2h_service=None, metadata_msg=None):
        """One step of lanes (step_message.Lane, in message order); request_id numbers lane 0's chunk."""
        lanes = [step_message.Lane(*lane) for lane in lanes]
        self.validate_step(lanes)
        self._check_cache(kv_cache)
        if not self.traces:
            raise RuntimeError("capture_trace must run before serving steps")
        if d2h_service is not self.d2h_service:
            raise ValueError("The D2H service must match the captured service")
        n = len(lanes)
        tokens = ttnn.reshape(input_tensor, (1, self.max_lanes * self.slab))
        if n < self.max_lanes:
            tokens = ttnn.slice(tokens, (0, 0), (1, n * self.slab))
        ttnn.copy(tokens, self.inputs[n])
        if metadata_msg is not None:
            ttnn.copy(metadata_msg, self.metadata)
        elif self.d2h_service is not None:
            raise ValueError("D2H acknowledgments require request metadata")
        self._stage(lanes)
        ttnn.execute_trace(self.mesh_device, self.traces[n], cq_id=0, blocking=False)
        ttnn.synchronize_device(self.mesh_device)
        for lane in lanes:
            self.slot_ends[lane.slot] = lane.end
        if self.layer_completion_sink is not None:
            for i in range(n):
                for layer_idx in range(self.config.num_layers):
                    self.layer_completion_sink(layer_idx, request_id + i)
        ttnn.deallocate(input_tensor)
        if metadata_msg is not None:
            ttnn.deallocate(metadata_msg)

    def prefill_chunk(self, *args, **kwargs):
        raise RuntimeError("The batched runtime serves step messages (prefill_step), not single chunks")

    def warmup_ack_count(self):
        return self.config.num_layers if self.d2h_service is not None else 0

    def release_trace(self):
        for trace_id in self.traces.values():
            ttnn.release_trace(self.mesh_device, trace_id)
        self.traces = {}
        attention_operations.PROJECTION_FIDELITY_OVERRIDE = self.saved_fidelity_override
        super().release_trace()
