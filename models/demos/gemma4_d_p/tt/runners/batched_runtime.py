# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Packed traced prefill forwards into engine-owned Gemma4 KV slots: up to forward_chunks chunks of 2048 tokens each.

Implements prefill_forward for PREFILL_FORWARD_CHUNKS > 1 (forward_entries): each non-empty entry is one chunk of one
request, a request may take several adjacent entries, and every entry starts on a 256-token slab boundary
(Gemma4PrefillRuntime.resume_granularity), its tokens rotated to its block-cyclic rows
(chunk_layout.pack_forward_tokens). One trace per number of non-empty entries, all compiled before any is captured: a
compile pass allocates persistent tensors lazily, and allocated after another trace's capture they can land on that
trace's freed intermediates. Projections take the fidelity of one 2k chunk (HiFi2), not of the stacked rows.

Layer completions: a forward of n entries completes like n chunks in entry order, num_layers per entry, each naming
its entry's (slot, start, end), and none precedes its KV. With a D2H ack service, entry 0's records go per layer from
the trace and the other entries' after its last layer, each from that entry's own 12-byte metadata tensor (the
LayerAckDecoder needs a chunk's records contiguous on the rank); otherwise the host reports them all once the forward
finishes (set_layer_completion_sink).
"""

import torch

import ttnn
from models.demos.common.prefill.forward_entries import PrefillEntry
from models.demos.gemma4_d_p.tt.common import create_tt_model
from models.demos.gemma4_d_p.tt.prefill_metadata import PrefillLanes
from models.demos.gemma4_d_p.tt.runners.runtime import Gemma4PrefillRuntime


class Gemma4BatchedPrefillRuntime(Gemma4PrefillRuntime):
    def __init__(self, *, mesh_device, hf_model_id, tt_cache_path, config):
        super().__init__(mesh_device=mesh_device, hf_model_id=hf_model_id, tt_cache_path=tt_cache_path, config=config)
        self.forward_chunks = config.forward_chunks
        self.entry_counts = range(self.forward_chunks, 0, -1)  # widest first
        self.slab = config.chunk_size // self.mesh_config.cp_degree
        self.inputs = {}
        self.positions = {}
        self.traces = {}
        self.outputs = {}
        self.entry_metadata = []

    def _host_rows(self, seqs):
        """Per-entry chunk-long sequences, rotated -> the [1, total] host tensor whose CP shard is each rank's slabs of
        every entry in entry order: the packed payload (chunk_layout.pack_forward_tokens)."""
        cp, n = self.mesh_config.cp_degree, len(seqs)
        rows = torch.stack([torch.as_tensor(s, dtype=torch.int64).view(cp, self.slab) for s in seqs], dim=1)
        return ttnn.from_torch(
            rows.reshape(1, cp * n * self.slab).to(torch.int32).contiguous(),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, self.config.mesh_shape, dims=(1, None)),
        )

    def _host_entry_metadata(self, entry):
        return ttnn.from_torch(
            torch.tensor([entry.slot, entry.start, entry.end], dtype=torch.int64).reshape(1, 1, 1, 3),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _zero_entries(self, n):
        chunk = self.config.chunk_size
        return [PrefillEntry(0, k * chunk, (k + 1) * chunk) for k in range(n)]

    def _stage(self, entries):
        """Host refresh of the positions, per-entry model metadata and per-entry ack records a replay reads."""
        n = len(entries)
        positions = [self._chunk_positions(e.start) for e in entries]
        ttnn.copy_host_to_device_tensor(self._host_rows(positions), self.positions[n])
        for metadata, e in zip(self.model.lane_prefill_metadata(n), entries):
            metadata.update(slot_idx=e.slot, kv_actual_global=e.start)
        if n > 1:
            self.model.lane_vector_metadata(n).update(
                slot_idx=[e.slot for e in entries], kv_actual_global=[e.start for e in entries]
            )
        for record, e in zip(self.entry_metadata, entries):
            ttnn.copy_host_to_device_tensor(self._host_entry_metadata(e), record)

    def _forward_entries(self, n):
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
            metadata_msg=self.entry_metadata[0],
        )
        if self.d2h_service is not None:
            for record in self.entry_metadata[1:n]:
                for _ in range(self.config.num_layers):
                    ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(self.d2h_service, metadata=record)
        return output

    def _run_once(self, n):
        """An untraced n-entry forward on slot 0 (compiles its programs)."""
        self._stage(self._zero_entries(n))
        output = self._forward_entries(n)
        ttnn.synchronize_device(self.mesh_device)
        output.deallocate(True)

    def compile(self, kv_cache):
        _, self.model, _, _ = create_tt_model(
            mesh_config=self.mesh_config,
            hf_model_id=self.hf_model_id,
            prefill_chunk_size=self.config.chunk_size,
            max_seq_len=self.config.max_seq_len,
            max_batch_size=self.config.num_users,
            ring_kv_caches=kv_cache,
            tt_cache_path=self.tt_cache_path,
        )
        # Every entry's metadata, inputs, positions and ack record before any activation is live.
        self.model.lane_prefill_metadata(self.forward_chunks)
        for n in self.entry_counts:
            if n > 1:
                self.model.lane_vector_metadata(n)
            zeros = [torch.zeros(self.config.chunk_size, dtype=torch.int32)] * n
            self.inputs[n] = ttnn.to_device(self._host_rows(zeros), self.mesh_device)
            self.positions[n] = ttnn.to_device(self._host_rows(zeros), self.mesh_device)
        self.entry_metadata = [
            ttnn.to_device(self._host_entry_metadata(e), self.mesh_device)
            for e in self._zero_entries(self.forward_chunks)
        ]
        for n in self.entry_counts:
            self._run_once(n)

    def capture_trace(self, kv_cache):
        self._check_cache(kv_cache)
        if self.d2h_service is not None:
            # Compiles the ack op (warmup_ack_count records); every other program is compiled already.
            self._run_once(1)
        for n in self.entry_counts:
            self._stage(self._zero_entries(n))
            trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
            self.outputs[n] = self._forward_entries(n)
            ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(self.mesh_device)
            self.traces[n] = trace_id

    def validate_forward(self, entries):
        """Gemma4's rules on top of forward_entries.validate_forward_entries (which the runner applies): every
        non-empty entry starts at a multiple of the resume granularity, and a slot's first entry starts within what the
        slot holds."""
        chunk = self.config.chunk_size
        real = [e for e in entries if not e.is_empty]
        if len(entries) != self.forward_chunks:
            raise ValueError(f"A forward holds {self.forward_chunks} entries, got {len(entries)}")
        if not real or entries[: len(real)] != real:
            raise ValueError(f"A forward needs 1 to {self.forward_chunks} non-empty entries before any empty one")
        seen = set()
        for i, e in enumerate(real):
            if not 0 <= e.slot < self.config.num_users:
                raise ValueError(f"Entry {i}: KV slot {e.slot} is outside [0, {self.config.num_users})")
            if e.start < 0 or e.start % self.resume_granularity:
                raise ValueError(
                    f"Entry {i}: start {e.start} is not a nonnegative multiple of {self.resume_granularity}"
                )
            if not e.start < e.end <= min(e.start + chunk, self.config.max_seq_len):
                raise ValueError(f"Entry {i}: [{e.start}, {e.end}) must hold 1 to {chunk} tokens within the context")
            previous = real[i - 1] if i else None
            if previous is not None and previous.slot == e.slot:
                if previous.end != previous.start + chunk or e.start != previous.end:
                    raise ValueError(
                        f"Entry {i}: slot {e.slot}'s entries must be consecutive full chunks, partial last"
                    )
            elif e.slot in seen:
                raise ValueError(f"Entry {i}: slot {e.slot}'s entries must be adjacent")
            elif e.start > self.slot_ends[e.slot]:  # a resume may rewind into the cached prefix, not skip past it
                raise ValueError(f"Slot {e.slot} holds {self.slot_ends[e.slot]} tokens, got an entry at {e.start}")
            seen.add(e.slot)
        return real

    def prefill_forward(self, input_tensor, kv_cache, entries, request_id=0, d2h_service=None, *, metadata_msg=None):
        """One packed forward (forward_entries.PrefillEntry, in pack order, empty ones at the tail). Frees
        input_tensor and metadata_msg after the forward; if it raises, they stay the caller's, as in prefill_chunk."""
        real = self.validate_forward(list(entries))
        self._check_cache(kv_cache)
        if not self.traces:
            raise RuntimeError("capture_trace must run before serving forwards")
        if d2h_service is not self.d2h_service:
            raise ValueError("The D2H service must match the captured service")
        if metadata_msg is None and self.d2h_service is not None:
            raise ValueError("D2H acknowledgments require request metadata")
        n = len(real)
        tokens = ttnn.reshape(input_tensor, (1, self.forward_chunks * self.slab))
        if n < self.forward_chunks:
            tokens = ttnn.slice(tokens, (0, 0), (1, n * self.slab))
        ttnn.copy(tokens, self.inputs[n])
        self._stage(real)
        ttnn.execute_trace(self.mesh_device, self.traces[n], cq_id=0, blocking=False)
        ttnn.synchronize_device(self.mesh_device)
        for e in real:
            self.slot_ends[e.slot] = e.end
        if self.layer_completion_sink is not None:
            for e in real:
                for layer_idx in range(self.config.num_layers):
                    self.layer_completion_sink.layers_completed(layer_idx, layer_idx + 1, e.slot, e.start, e.end)
        ttnn.deallocate(input_tensor)
        if metadata_msg is not None:
            ttnn.deallocate(metadata_msg)

    def prefill_chunk(self, *args, **kwargs):
        raise RuntimeError("The batched runtime serves packed forwards (prefill_forward), not single chunks")

    def release_trace(self):
        for trace_id in self.traces.values():
            ttnn.release_trace(self.mesh_device, trace_id)
        self.traces = {}
        super().release_trace()
