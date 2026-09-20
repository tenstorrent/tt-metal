# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Adapt rotated Q to the aligned-group contract of sliding ring SDPA."""

from enum import Enum
from types import SimpleNamespace

import torch

import ttnn


class SlidingChunkMode(Enum):
    ALIGNED = "aligned"
    SINGLE_GROUP = "single_group"
    TWO_GROUPS = "two_groups"


class SlidingChunk:
    """Stable trace inputs for native or reordered sliding attention."""

    def __init__(self, mesh_config, chunk_size, max_seq_len):
        self.mesh_config = mesh_config
        self.chunk_size = chunk_size
        self.max_seq_len = max_seq_len
        self._buffers = {}
        self._expanded_indices = {}

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

    def select_mode(self, actual_start, actual_end, mode=None):
        """Choose the cheapest graph covering the real queries; validate capture overrides."""
        group_end = (actual_start // self.chunk_size + 1) * self.chunk_size
        local = self.chunk_size // self.mesh_config.cp_degree
        if actual_start % local == 0 and actual_end <= group_end:
            required = SlidingChunkMode.ALIGNED
        elif actual_end <= group_end:
            required = SlidingChunkMode.SINGLE_GROUP
        else:
            required = SlidingChunkMode.TWO_GROUPS
        if mode is None:
            return required
        if mode == SlidingChunkMode.ALIGNED and required != SlidingChunkMode.ALIGNED:
            raise ValueError("The aligned SWA trace requires a CP-block-aligned start and queries within one group")
        if mode == SlidingChunkMode.SINGLE_GROUP and required == SlidingChunkMode.TWO_GROUPS:
            raise ValueError("The single-group SWA trace cannot span two chunk groups")
        return mode

    def update(self, actual_start, positions, *, mode):
        """Stage only the groups used by the selected SWA graph."""
        self.mode = mode
        num_groups = 2 if mode == SlidingChunkMode.TWO_GROUPS else 1
        group_start = actual_start // self.chunk_size * self.chunk_size
        self.group_starts = []
        for group in range(num_groups):
            origin = min(group_start + group * self.chunk_size, self.max_seq_len - self.chunk_size)
            self.group_starts.append(self._stage(f"start{group}", torch.tensor([origin]).reshape(1, 1, 1, 1)))
        if mode == SlidingChunkMode.ALIGNED:
            return
        local = self.chunk_size // self.mesh_config.cp_degree
        # Cache-local rows relative to the first group, in request order.
        rows = positions // self.chunk_size * local + positions % local - group_start // self.mesh_config.cp_degree
        rank_rows = torch.arange(local).expand(self.mesh_config.cp_degree, -1)
        for group in range(num_groups):
            origin = min(group_start + group * self.chunk_size, self.max_seq_len - self.chunk_size)
            group_offset = (origin - group_start) // self.mesh_config.cp_degree
            indices = (rank_rows + group_offset - rows[:, :1]).clamp(0, local - 1)
            self._stage(f"q{group}", indices.reshape(1, 1, -1, 1), seq_dim=2)
        # Padded rows outside the selected groups have discarded outputs.
        rows = rows.clamp(max=num_groups * local - 1)
        self._stage("output", rows.reshape(1, 1, -1, 1), seq_dim=2)
        for name, heads, width in self._expanded_indices:
            if name != "q1" or num_groups == 2:
                self._expand_indices(name, heads, width)

    def _expand_indices(self, name, heads, width):
        key = (name, heads, width)
        self._expanded_indices[key] = ttnn.repeat(
            self._buffers[name],
            ttnn.Shape((1, heads, 1, width)),
            optional_output_tensor=self._expanded_indices.get(key),
        )

    def _gather_rows(self, tensor, name):
        key = (name, tensor.shape[1], tensor.shape[3])
        if key not in self._expanded_indices:
            # Allocate during warmup; update shared indices before subsequent replays.
            self._expand_indices(*key)
        return ttnn.gather(tensor, dim=2, index=self._expanded_indices[key], memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def attention(self, tt_q, prefill_metadata, attention_fn, **kwargs):
        """Run one native call or reorder Q across its one or two groups."""
        if self.mode == SlidingChunkMode.ALIGNED:
            metadata = SimpleNamespace(slot_idx=prefill_metadata.slot_idx, kv_actual_global=self.group_starts[0])
            return attention_fn(tt_q=tt_q, prefill_metadata=metadata, **kwargs)
        outputs = []
        for group, start in enumerate(self.group_starts):
            query = self._gather_rows(tt_q, f"q{group}")
            metadata = SimpleNamespace(slot_idx=prefill_metadata.slot_idx, kv_actual_global=start)
            outputs.append(attention_fn(tt_q=query, prefill_metadata=metadata, **kwargs))
            query.deallocate(True)
        if len(outputs) == 1:
            combined = outputs[0]
        else:
            combined = ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for tensor in outputs:
                tensor.deallocate(True)
        result = self._gather_rows(combined, "output")
        combined.deallocate(True)
        return result
