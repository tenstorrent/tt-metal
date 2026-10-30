"""Opt-in prefill positions using original tables and existing native operations.

Constructed once per layer kind in Gemma4Model.shared_setup, never globally.
One model/mesh and ordered default CQ0 caller owns every index buffer. A refresh
follows earlier readers and precedes both new table readers on CQ0. Gathered
tables and SDPA outputs are local temporaries, not retained by this owner.
This module does not prepare programs or change the trace-retirement contract.
"""

import torch
import ttnn


class RuntimePrefillPositions:
    def __init__(self, mesh, *, maximum_sequence, full_attention):
        if int(ttnn.get_current_command_queue_id_for_thread()) != 0:
            raise ValueError("Runtime prefill positions require one ordered CQ0 caller")
        if maximum_sequence <= 0 or maximum_sequence % 128 or maximum_sequence > 6656:
            raise ValueError("Expected the original page-aligned native chunk bound")
        self.mesh = mesh
        self.maximum_sequence = maximum_sequence
        self.full_attention = full_attention
        self.rope_indices = {
            sequence: self._upload(torch.zeros((1, sequence), dtype=torch.int32), ttnn.uint32)
            for sequence in range(128, maximum_sequence + 1, 128)
        }
        self.start = self._upload(torch.zeros(1, dtype=torch.int32), ttnn.int32) if full_attention else None
        self.initial_buffer_ids = self.buffer_ids()

    def _upload(self, values, dtype):
        return ttnn.from_torch(
            values.contiguous(), device=self.mesh, dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    @staticmethod
    def _refresh(destination, values, dtype):
        if int(ttnn.get_current_command_queue_id_for_thread()) != 0:
            raise ValueError("Runtime prefill positions require one ordered CQ0 caller")
        # The source has host storage only. This does not replace the device root.
        source = ttnn.from_torch(values.contiguous(), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(source, destination, cq_id=0)

    def buffer_ids(self):
        result = {f"rope_Q{sequence}": value.buffer_unique_id() for sequence, value in self.rope_indices.items()}
        if self.start is not None:
            result["full_start"] = self.start.buffer_unique_id()
        return result

    def rope_tables(self, layer, position, sequence):
        if sequence not in self.rope_indices:
            raise ValueError(f"Unprepared RoPE width {sequence}; no lazy persistent allocation")
        if not 0 <= position < layer.max_context or sequence > layer.max_context:
            raise ValueError("RoPE position/width outside the original table")
        available = min(sequence, layer.max_context - position)
        positions = torch.cat((torch.arange(position, position + available, dtype=torch.int32),
                               torch.arange(sequence - available, dtype=torch.int32))).reshape(1, sequence)
        index = self.rope_indices[sequence]
        self._refresh(index, positions, ttnn.uint32)
        shape = ttnn.Shape([1, 1, sequence, layer.head_dim])
        result = []
        for table in (layer.decode_cos, layer.decode_sin):
            rows = ttnn.embedding(index, table, layout=ttnn.TILE_LAYOUT,
                                  memory_config=ttnn.DRAM_MEMORY_CONFIG)
            result.append(ttnn.reshape(rows, shape, shape))
        return tuple(result)

    def full_sdpa(self, layer, query, keys, values, page_table, *, position, program_config):
        if not self.full_attention or layer.sliding or position < 0 or position % 128:
            raise ValueError("Expected the original page-aligned full-attention start")
        expected = layer.kv_dtypes
        supported = ((ttnn.bfloat8_b, ttnn.bfloat8_b),
                     (ttnn.bfloat8_b, ttnn.bfloat4_b),
                     (ttnn.bfloat4_b, ttnn.bfloat4_b))
        if (expected not in supported or query.dtype != ttnn.bfloat8_b
                or (keys.dtype, values.dtype) != expected):
            raise ValueError("Full-attention requires BF8 Q and the exact selected K/V policy")
        # Config selection remains in Decoder and still sees the scalar position.
        # The native tensor contains one shared group start, never per-B positions.
        self._refresh(self.start, torch.tensor([position], dtype=torch.int32), ttnn.int32)
        return ttnn.transformer.chunked_scaled_dot_product_attention(
            query, keys, values, page_table, chunk_start_idx_tensor=self.start,
            scale=1.0, program_config=program_config, compute_kernel_config=layer.pref_compute,
        )
