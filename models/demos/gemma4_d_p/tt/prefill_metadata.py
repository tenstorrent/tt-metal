# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-resident request metadata shared by prefill layers."""

import torch

import ttnn


def chunk_positions(actual_start, chunk_size, cp):
    """Absolute positions in the cache writer's CP-rank and local-row order."""
    local = chunk_size // cp
    group, remainder = divmod(actual_start, chunk_size)
    boundary_rank, offset = divmod(remainder, local)
    ranks = torch.arange(cp).unsqueeze(1)
    first_row = group * local + (ranks < boundary_rank) * local + (ranks == boundary_rank) * offset
    rows = first_row + torch.arange(local)
    return rows // local * chunk_size + ranks * local + rows % local


class PrefillMetadata:
    """Request tensors whose addresses remain stable across trace replays."""

    def __init__(self, mesh_config, chunk_size, max_seq_len, num_users=1):
        self.mesh_config = mesh_config
        self.chunk_size = chunk_size
        self.max_seq_len = max_seq_len
        self.num_users = num_users
        self._buffers = {}
        self.update(slot_idx=0, actual_start=0, actual_end=min(chunk_size, max_seq_len))

    def _stage(self, name, values, seq_dim=None):
        mesh_device = self.mesh_config.device
        dims = [None, None]
        dims[self.mesh_config.cp_axis] = seq_dim
        host = ttnn.from_torch(
            values,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=dims),
        )
        if name not in self._buffers:
            self._buffers[name] = ttnn.to_device(host, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            ttnn.copy_host_to_device_tensor(host, self._buffers[name])
        return self._buffers[name]

    def update(self, *, slot_idx, actual_start, actual_end):
        """Stage request bounds and RoPE positions before replay."""
        if not 0 <= slot_idx < self.num_users:
            raise ValueError(f"slot_idx must be in [0, {self.num_users}), got {slot_idx}")
        if actual_start < 0 or actual_start % ttnn.TILE_SIZE:
            raise ValueError(f"actual_start must be nonnegative and 32-token aligned, got {actual_start}")
        if not actual_start < actual_end <= min(actual_start + self.chunk_size, self.max_seq_len):
            raise ValueError("require actual_start < actual_end <= min(actual_start + chunk_size, max_seq_len)")
        self.slot_idx = self._stage("slot", torch.tensor([slot_idx]).reshape(1, 1, 1, 1))
        self.kv_actual_global = self._stage("start", torch.tensor([actual_start]).reshape(1, 1, 1, 1))
        self.actual_end = self._stage("end", torch.tensor([actual_end]).reshape(1, 1, 1, 1))
        positions = chunk_positions(actual_start, self.chunk_size, self.mesh_config.cp_degree)
        # Padded rows can extend beyond the RoPE table; their output is discarded.
        safe_positions = positions.masked_fill(positions >= self.max_seq_len, 0)
        self.positions = self._stage("positions", safe_positions.reshape(1, -1), seq_dim=1)
