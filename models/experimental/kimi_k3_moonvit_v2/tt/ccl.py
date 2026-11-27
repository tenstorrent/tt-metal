# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Ring setup for sequence parallelism.

`ParallelFactor` names how many devices a dimension is split over and along which mesh axis.
`TT_CCL` owns what ring attention needs across calls: `num_links`, the topology, and per slot a set of
global semaphores and a pair of persistent K/V gather buffers. Semaphores are hardware resources, so
one `TT_CCL` is built per mesh and shared by every layer.

Why slots: `ring_joint_scaled_dot_product_attention` has no start barrier. A device's K/V all-gather
writes its shard into its neighbours' gather buffers and increments their semaphores as soon as the
op starts, and each device zeroes its own semaphores when it finishes. A device that is one call
ahead of a neighbour would overwrite K/V the neighbour is still reading, and could have its
increments wiped by the neighbour's end-of-op reset (the neighbour then hangs on the next call). If
consecutive calls use different slots this cannot happen: to start the call after next, a device
must have finished the next call, which needs every neighbour's shard for it, so every neighbour has
finished the current call.

Each layer's slot is fixed when the model is built (`ccl_slot_for_layer`), not picked per call: a trace
records the slot of each call, so a slot that depended on how many calls ran before would differ
between traced and untraced runs, and an odd number of calls per trace would put the last call of
one replay and the first of the next on the same slot.
"""

from __future__ import annotations

from typing import NamedTuple

import ttnn

# Semaphores per ring attention call: one per all-gather direction (forward, backward).
RING_ATTENTION_SEMAPHORES = 2
# Slots for the encoder's layers: alternating 0, 1, and the last layer on 2 when the count is odd.
CCL_SLOTS = 3


def ccl_slot_for_layer(layer: int, num_layers: int) -> int:
    """The slot of `layer` out of `num_layers` ring attention calls per forward, such that any two
    consecutive calls use different slots, including the last call of one forward and the first of
    the next. A single call per forward always uses slot 0: back-to-back forwards then share it."""
    if not 0 <= layer < num_layers:
        raise ValueError(f"layer {layer} not in [0, {num_layers})")
    if num_layers % 2 == 1 and num_layers > 1 and layer == num_layers - 1:
        return 2
    return layer % 2


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
        self._semaphores: dict[int, list[ttnn.GlobalSemaphore]] = {}
        self._buffers: dict[tuple, tuple[ttnn.Tensor, ttnn.Tensor]] = {}

    def ring_attention_semaphores(self, slot: int) -> list[ttnn.GlobalSemaphore]:
        """`RING_ATTENTION_SEMAPHORES` global semaphores for `ring_joint_scaled_dot_product_attention`
        in `slot`, created on first use."""
        _check_slot(slot)
        if slot not in self._semaphores:
            grid = self.mesh_device.compute_with_storage_grid_size()
            cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
            self._semaphores[slot] = [
                ttnn.create_global_semaphore(self.mesh_device, cores, 0) for _ in range(RING_ATTENTION_SEMAPHORES)
            ]
            # Every device must have zeroed its semaphores before any device starts the op, or an
            # increment from a fast neighbour lands first and is overwritten.
            ttnn.synchronize_device(self.mesh_device)
        return self._semaphores[slot]

    def ring_attention_buffers(self, slot: int, shape, dim: int, mesh_axis: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Persistent (K, V) gather buffers in `slot` for K and V shards of `shape`, gathered along
        `dim` over `mesh_axis`; allocated once per slot and shape.

        Allocated uninitialized: the gather overwrites all of it, so no zero fill is needed.
        """
        _check_slot(slot)
        gathered = list(shape)
        gathered[dim] = shape[dim] * tuple(self.mesh_device.shape)[mesh_axis]
        key = (slot, tuple(gathered))
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


def _check_slot(slot: int) -> None:
    if not 0 <= slot < CCL_SLOTS:
        raise ValueError(f"CCL slot {slot} not in [0, {CCL_SLOTS})")
