# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A reduce-scatter running concurrently with the owned kernels (iteration 47).

ttnn serializes programs on one command queue, but programs on different sub-devices only wait for their own
sub-device's earlier programs. :class:`CollectiveOverlap` holds a sub-device manager that splits the chip into a
compute part (the first ``grid.y - ccl_rows`` core rows) and a collective part (the last ``ccl_rows`` rows, where the
async reduce-scatter's workers and semaphores live). A consumer that wants its reduce-scatter hidden enters the
window (the split manager loaded), issues the reduce-scatter on the collective sub-device, runs its independent
work on the compute rows (every program inside the window must fit one sub-device: the owned ops take
``grid_rows``), and leaves the window (the default manager back; a switch pair costs ~7 us, outputs are exact across
switches: ``doc/optimization/evidence/iter47_subdevice_switch_scoping2.log``).

The global semaphores must be created with the split manager loaded (created under the default manager they work
but nothing overlaps: ``iter47_subdevice_switch_scoping.log``), and consecutive reduce-scatters must not share a
semaphore set (ttml's own collectives rotate ``kNumSemaphoresPairs`` sets). The reduce-scatter's staging buffers
must be persistent: the op frees its intermediate and penult staging buffers on the host right after the enqueue
(``reduce_scatter_minimal_async.cpp``), which is safe only when the next program waits for it; inside the window
the products on the other sub-device get those addresses and run concurrently, and the 8-layer TP=4 loss match
against torch failed by 0.62-0.73 while every serial control passed (``iter48_loss_comparison_tp4_*.txt``). The
persistent output is cloned after the window so the autograd graph never holds the window's buffer.
"""

from __future__ import annotations

import ttnn

__all__ = ["CollectiveOverlap"]


class CollectiveOverlap:
    def __init__(
        self,
        device,
        ccl_rows: int = 3,
        workers_per_link: int = 4,
        num_links: int = 2,
        serial: bool = False,
        semaphore_sets: int = 4,
        serial_async: bool = False,
    ):
        """``serial`` (a control, review round 14): the window's routes without sub-devices: no manager, the
        reduce-scatter through ttml's plain collective, the products on the full grid. ``serial_async``: the same
        but through this class's own async reduce-scatter call (its parameters and semaphore sets) on all cores."""
        self.serial = serial or serial_async
        self.serial_async = serial_async
        self.windows = 0
        self.active = False
        self.tp = 4
        grid = device.compute_with_storage_grid_size()
        if serial_async:
            self.device = device
            self.compute_rows = 0
            self.num_links, self.workers_per_link, self.ccl_id = num_links, None, None
            crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
            self.semaphore_sets = [
                (
                    [ttnn.create_global_semaphore(device, crs, 0) for _ in range(3)],
                    ttnn.create_global_semaphore(device, crs, 0),
                )
                for _ in range(semaphore_sets)
            ]
            self.next_set = 0
            return
        if serial:
            self.compute_rows = 0
            return
        assert 0 < ccl_rows < grid.y, "the collective rows must leave compute rows"
        self.device = device
        self.compute_rows = grid.y - ccl_rows
        self.num_links = num_links
        self.workers_per_link = workers_per_link  # 8 per link does not fit 33 cores (bad_alloc)
        compute_crs = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, self.compute_rows - 1))]
        )
        self.ccl_crs = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, self.compute_rows), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
        )
        self.split = device.create_sub_device_manager(
            [ttnn.SubDevice([compute_crs]), ttnn.SubDevice([self.ccl_crs])], 0
        )
        self.stall_group = [ttnn.SubDeviceId(0), ttnn.SubDeviceId(1)]
        self.ccl_id = ttnn.SubDeviceId(1)
        self.active = False
        device.load_sub_device_manager(self.split)
        self.semaphore_sets = [
            (
                [ttnn.create_global_semaphore(device, self.ccl_crs, 0) for _ in range(3)],
                ttnn.create_global_semaphore(device, self.ccl_crs, 0),
            )
            for _ in range(semaphore_sets)
        ]
        self.next_set = 0
        device.clear_loaded_sub_device_manager()
        self.buffer_sets = {}  # (shape, dim, cluster_axis) -> [(intermediate, output, penult?), ...] persistent

    def _persistent_buffers(self, tensor, dim: int, cluster_axis: int, sets: int = 2):
        key = (tuple(int(v) for v in tensor.shape), dim, cluster_axis)
        if key not in self.buffer_sets:
            import torch

            out_shape = list(key[0])
            out_shape[dim] //= 4 if cluster_axis is None else self.tp
            allocated = []
            for _ in range(sets):
                staging = ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(
                    tensor, dim, topology=ttnn.Topology.Ring, cluster_axis=cluster_axis
                )
                out = ttnn.from_torch(
                    torch.zeros(out_shape, dtype=torch.bfloat16),
                    dtype=tensor.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
                )
                allocated.append([staging[0], out] + list(staging[1:]))
            self.buffer_sets[key] = [allocated, 0]
        sets_, i = self.buffer_sets[key]
        self.buffer_sets[key][1] = (i + 1) % len(sets_)
        return sets_[i]

    def enter(self) -> int:
        """Load the split manager; returns the compute rows the window's owned ops must take as ``grid_rows``."""
        assert not self.active, "collective overlap windows do not nest"
        if self.serial:
            self.active = True
            self.windows += 1
            return 0
        self.device.load_sub_device_manager(self.split)
        self.device.set_sub_device_stall_group(self.stall_group)
        self.active = True
        self.windows += 1
        return self.compute_rows

    def exit(self) -> None:
        assert self.active
        if self.serial:
            self.active = False
            return
        self.device.reset_sub_device_stall_group()
        self.device.clear_loaded_sub_device_manager()
        self.active = False

    def reduce_scatter(self, tensor, dim: int, cluster_axis: int):
        """The async reduce-scatter on the collective sub-device (inside a window)."""
        assert self.active, "reduce_scatter outside a collective overlap window"
        if self.serial and not self.serial_async:
            import ttml

            return ttml.core.distributed.reduce_scatter(tensor, dim, cluster_axis)
        semaphores, barrier = self.semaphore_sets[self.next_set]  # a fresh set per call (see the module docstring)
        self.next_set = (self.next_set + 1) % len(self.semaphore_sets)
        persistent = None if self.serial_async else self._persistent_buffers(tensor, dim, cluster_axis)
        return ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            persistent,
            dim,
            semaphores,
            barrier_semaphore=barrier,
            num_links=self.num_links,
            topology=ttnn.Topology.Ring,
            subdevice_id=self.ccl_id,
            cluster_axis=cluster_axis,
            num_workers_per_link=self.workers_per_link,
        )
