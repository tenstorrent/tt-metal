# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Adapt rotated Q to the aligned-group contract of sliding ring SDPA."""

from types import SimpleNamespace

import torch

import ttnn


class SlidingChunk:
    """Stable trace inputs for two aligned SWA calls and restoring request order."""

    def __init__(self, mesh_config, chunk_size, max_seq_len):
        self.mesh_config = mesh_config
        self.chunk_size = chunk_size
        self.max_seq_len = max_seq_len
        self._buffers = {}

    def _stage(self, name, values, seq_dim=None):
        mesh_device = self.mesh_config.device
        dims = [None, None]
        dims[self.mesh_config.cp_axis] = seq_dim
        host = ttnn.from_torch(
            values,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT if seq_dim is None else ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=dims),
        )
        if name not in self._buffers:
            self._buffers[name] = ttnn.to_device(host, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            ttnn.copy_host_to_device_tensor(host, self._buffers[name])
        return self._buffers[name]

    def update(self, actual_start, positions):
        """Stage the two group origins and local gather indices before replay."""
        local = self.chunk_size // self.mesh_config.cp_degree
        group_start = actual_start // self.chunk_size * self.chunk_size
        # Cache-local rows relative to the first group, in request order.
        rows = positions // self.chunk_size * local + positions % local - group_start // self.mesh_config.cp_degree
        rank_rows = torch.arange(local).expand(self.mesh_config.cp_degree, -1)
        self.q_indices = []
        self.group_starts = []
        for group in range(2):
            origin = min(group_start + group * self.chunk_size, self.max_seq_len - self.chunk_size)
            group_offset = (origin - group_start) // self.mesh_config.cp_degree
            indices = (rank_rows + group_offset - rows[:, :1]).clamp(0, local - 1)
            self.q_indices.append(self._stage(f"q{group}", indices.reshape(1, 1, -1, 1), seq_dim=2))
            self.group_starts.append(self._stage(f"start{group}", torch.tensor([origin]).reshape(1, 1, 1, 1)))
        self.output_indices = self._stage("output", rows.reshape(1, 1, -1, 1), seq_dim=2)

    @staticmethod
    def _gather_rows(tensor, indices):
        indices = ttnn.repeat(indices, ttnn.Shape((1, tensor.shape[1], 1, tensor.shape[3])))
        result = ttnn.gather(tensor, dim=2, index=indices, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        indices.deallocate(True)
        return result

    def attention(self, tt_q, prefill_metadata, attention_fn, **kwargs):
        """Run aligned SWA twice, then restore the input's block-cyclic row order."""
        outputs = []
        for indices, start in zip(self.q_indices, self.group_starts):
            query = self._gather_rows(tt_q, indices)
            metadata = SimpleNamespace(slot_idx=prefill_metadata.slot_idx, kv_actual_global=start)
            outputs.append(attention_fn(tt_q=query, prefill_metadata=metadata, **kwargs))
            query.deallocate(True)
        combined = ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        result = self._gather_rows(combined, self.output_indices)
        for tensor in (*outputs, combined):
            tensor.deallocate(True)
        return result
