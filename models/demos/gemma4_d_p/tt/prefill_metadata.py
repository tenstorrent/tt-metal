# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-resident request metadata shared by prefill layers."""

import torch

import ttnn


class PrefillMetadata:
    """Slot and KV position tensors whose addresses remain stable across trace replays."""

    def __init__(self, mesh_config):
        self.mesh_config = mesh_config
        self.slot_idx = self._device_scalar()
        self.kv_actual_global = self._device_scalar()

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
        for tensor, value in ((self.slot_idx, slot_idx), (self.kv_actual_global, kv_actual_global)):
            ttnn.copy_host_to_device_tensor(self._host_scalar(value), tensor)


class PrefillLanes(list):
    """A batched step's per-request metadata plus each request's CP-local rows.

    The rows are static at trace capture, so a step with different per-request chunk widths is a different trace.
    A plain list of metadata means every request has the same width.
    """

    def __init__(self, metadata, rows):
        super().__init__(metadata)
        if len(rows) != len(self):
            raise ValueError(f"{len(self)} lanes but {len(rows)} row counts")
        self.rows = tuple(int(r) for r in rows)
