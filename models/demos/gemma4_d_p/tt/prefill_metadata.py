# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-resident request metadata shared by prefill layers."""

import torch

import ttnn


class PrefillMetadata:
    """Slot and KV position tensors whose addresses remain stable across trace replays.

    With ``circular_capacity`` set, ``wrapped_kv_actual_global`` holds ``kv_actual_global`` modulo that capacity:
    the offset where a bounded sliding cache stores the chunk. Ring SDPA still reads the absolute
    ``kv_actual_global``.
    """

    def __init__(self, mesh_config, circular_capacity=None):
        self.mesh_config = mesh_config
        self.circular_capacity = circular_capacity
        self.slot_idx = self._device_scalar()
        self.kv_actual_global = self._device_scalar()
        self.wrapped_kv_actual_global = self._device_scalar() if circular_capacity is not None else None

    def _host_scalar(self, value):
        return ttnn.from_torch(
            torch.tensor([value], dtype=torch.int64).reshape(1, 1, 1, 1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_config.device),
        )

    def _device_scalar(self):
        return ttnn.to_device(self._host_scalar(0), self.mesh_config.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def update(self, *, slot_idx, kv_actual_global):
        """Update the existing device buffers before executing or replaying a chunk."""
        updates = [(self.slot_idx, slot_idx), (self.kv_actual_global, kv_actual_global)]
        if self.wrapped_kv_actual_global is not None:
            updates.append((self.wrapped_kv_actual_global, kv_actual_global % self.circular_capacity))
        for tensor, value in updates:
            ttnn.copy_host_to_device_tensor(self._host_scalar(value), tensor)
