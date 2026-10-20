# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-resident request metadata shared by prefill layers."""

from dataclasses import dataclass

import torch

import ttnn


class PrefillMetadata:
    """Slot and KV position tensors whose addresses remain stable across trace replays."""

    def __init__(self, mesh_config):
        self.mesh_config = mesh_config
        self.slot_idx = self._device_scalar()
        self.kv_actual_global = self._device_scalar()
        self.valid_global = self._device_scalar()

    def _host_scalar(self, value):
        return ttnn.from_torch(
            torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_config.device),
        )

    def _device_scalar(self):
        return ttnn.to_device(self._host_scalar(0), self.mesh_config.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def update(self, *, slot_idx, kv_actual_global, valid_global=None):
        """Update the existing device buffers before executing or replaying a chunk."""
        values = [(self.slot_idx, slot_idx), (self.kv_actual_global, kv_actual_global)]
        if valid_global is not None:
            values.append((self.valid_global, valid_global))
        for tensor, value in values:
            ttnn.copy_host_to_device_tensor(self._host_scalar(value), tensor)


@dataclass(frozen=True)
class PrefillRequestMetadata:
    descriptors: ttnn.Tensor
    index: int


class PackedPrefillMetadata:
    def __init__(self, mesh_config, max_requests, chunk_size):
        self.mesh_config = mesh_config
        self.max_requests = max_requests
        self.chunk_size = chunk_size
        self.layout = None
        self.tensor = ttnn.to_device(
            self._host_table(torch.zeros((max_requests, 8), dtype=torch.int64)),
            mesh_config.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.requests = tuple(PrefillRequestMetadata(self.tensor, index) for index in range(max_requests))

    def _host_table(self, rows):
        return ttnn.from_torch(
            rows,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_config.device),
        )

    def update(self, requests):
        """Stage (cache slot, prefix tokens, valid tokens, allocated tokens) for one packed batch."""
        requests = tuple(requests)
        if not 1 <= len(requests) <= self.max_requests:
            raise ValueError("Packed request count exceeds metadata capacity or is empty")
        quantum = 32 * self.mesh_config.cp_degree
        rows = torch.zeros((self.max_requests, 8), dtype=torch.int64)
        offset = 0
        for index, (slot, prefix, valid, span) in enumerate(requests):
            if slot < 0 or prefix < 0 or prefix % 32 or not 0 < valid <= span or span % quantum:
                raise ValueError("Invalid packed request slot, tile-aligned prefix, valid length, or allocated span")
            if max(slot, prefix + valid) > 0xFFFFFFFF:
                raise ValueError("Packed request metadata must fit uint32")
            if offset + span > self.chunk_size:
                raise ValueError("Packed request spans exceed the chunk size")
            rows[index, :6] = torch.tensor([1, slot, prefix, valid, offset // quantum, span // quantum])
            offset += span
        ttnn.copy_host_to_device_tensor(self._host_table(rows), self.tensor)
        self.layout = tuple(span for _, _, _, span in requests)
