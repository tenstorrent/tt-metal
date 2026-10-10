# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-resident request metadata shared by prefill layers."""

import torch

import ttnn


class PrefillMetadata:
    """Slot and KV position tensors whose addresses remain stable across trace replays.

    num_lanes > 1 holds a batched step's requests as B-element tensors (element b per request), the form the lanes
    ring SDPA reads; update then takes one value per request.
    """

    def __init__(self, mesh_config, num_lanes=1):
        self.mesh_config = mesh_config
        self.num_lanes = num_lanes
        self.slot_idx = self._device_vector()
        self.kv_actual_global = self._device_vector()

    def _host_vector(self, values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int64).reshape(1, 1, 1, self.num_lanes),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_config.device),
        )

    def _device_vector(self):
        return ttnn.to_device(
            self._host_vector([0] * self.num_lanes), self.mesh_config.device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def update(self, *, slot_idx, kv_actual_global):
        """Update the existing device buffers before executing or replaying a chunk (lists for several lanes)."""
        for tensor, value in ((self.slot_idx, slot_idx), (self.kv_actual_global, kv_actual_global)):
            values = list(value) if self.num_lanes > 1 else [value]
            ttnn.copy_host_to_device_tensor(self._host_vector(values), tensor)


class PrefillLanes(list):
    """A batched step's per-request metadata plus each request's CP-local rows.

    The rows are static at trace capture, so a step with different per-request chunk widths is a different trace.
    A plain list of metadata means every request has the same width. vector: the same requests' slots and prefixes
    as one PrefillMetadata with num_lanes elements, for the lanes ring SDPA.
    """

    def __init__(self, metadata, rows, vector=None):
        super().__init__(metadata)
        if len(rows) != len(self):
            raise ValueError(f"{len(self)} lanes but {len(rows)} row counts")
        self.rows = tuple(int(r) for r in rows)
        self.vector = vector
