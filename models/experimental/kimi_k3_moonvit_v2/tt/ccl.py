# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Ring setup for sequence parallelism.

`ParallelFactor` names how many devices a dimension is split over and along which mesh axis.
`TT_CCL` owns what ring attention needs across calls: `num_links`, the topology, the global
semaphores and the persistent K/V gather buffers. Semaphores are hardware resources, so one
`TT_CCL` is built per mesh and shared by every layer.

The semaphores alternate between two sets on consecutive ring attention calls, as tt_dit's
`CCLManager` does (the pattern is copied, not imported). The K and V gather buffers are one fixed
pair per gathered shape, shared by every layer: they are scratch inside the op, never its output.
"""

from __future__ import annotations

from typing import NamedTuple

import ttnn

# Semaphores per ring attention call: one per all-gather direction (forward, backward).
RING_ATTENTION_SEMAPHORES = 2


class ParallelFactor(NamedTuple):
    factor: int
    mesh_axis: int


class TT_CCL:
    def __init__(self, mesh_device: ttnn.MeshDevice, *, num_links: int, topology: ttnn.Topology) -> None:
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology

        # Sub-device 0 is the whole worker grid; no sub-device manager is loaded.
        self.sub_device_id = ttnn.SubDeviceId(0)
        self._semaphores: list[ttnn.GlobalSemaphore] | None = None
        self._semaphore_idx = 0
        self._buffers: dict[tuple, tuple[ttnn.Tensor, ttnn.Tensor]] = {}

    def ring_attention_semaphores(self) -> list[ttnn.GlobalSemaphore]:
        """Global semaphores for `ring_joint_scaled_dot_product_attention`, created once: two sets of
        `RING_ATTENTION_SEMAPHORES`, alternating per call."""
        # TODO: Settle whether the alternation is needed. A trace records the set each call used, and
        # with 27 layers (odd) the last call of one replay and the first of the next share set 0.
        # Either the semaphores must be reset before each replay, or alternating is not needed at all:
        # gemma4_d_p, gpt_oss_d_p and deepseek_v3_d_p pass one fixed set to every ring attention call.
        if self._semaphores is None:
            grid = self.mesh_device.compute_with_storage_grid_size()
            cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
            self._semaphores = [
                ttnn.create_global_semaphore(self.mesh_device, cores, 0) for _ in range(2 * RING_ATTENTION_SEMAPHORES)
            ]
        idx = self._semaphore_idx
        self._semaphore_idx = 1 - idx
        return self._semaphores[idx * RING_ATTENTION_SEMAPHORES : (idx + 1) * RING_ATTENTION_SEMAPHORES]

    def ring_attention_buffers(self, shape, dim: int, mesh_axis: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Persistent (K, V) gather buffers for K and V shards of `shape`, gathered along `dim` over
        `mesh_axis`; allocated once per shape and shared by every layer.

        Allocated uninitialized: the gather overwrites all of it, so no zero fill is needed.
        """
        gathered = list(shape)
        gathered[dim] = shape[dim] * tuple(self.mesh_device.shape)[mesh_axis]
        key = tuple(gathered)
        if key not in self._buffers:
            ttnn.synchronize_device(self.mesh_device)
            self._buffers[key] = tuple(
                ttnn.allocate_tensor_on_device(
                    ttnn.Shape(gathered), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh_device, ttnn.DRAM_MEMORY_CONFIG
                )
                for _ in range(2)
            )
            ttnn.synchronize_device(self.mesh_device)
        return self._buffers[key]
