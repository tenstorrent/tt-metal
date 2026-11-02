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
but nothing overlaps: ``iter47_subdevice_switch_scoping.log``).
"""

from __future__ import annotations

import ttnn

__all__ = ["CollectiveOverlap"]


class CollectiveOverlap:
    def __init__(self, device, ccl_rows: int = 3, workers_per_link: int = 4, num_links: int = 2):
        grid = device.compute_with_storage_grid_size()
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
        self.semaphores = [ttnn.create_global_semaphore(device, self.ccl_crs, 0) for _ in range(3)]
        self.barrier = ttnn.create_global_semaphore(device, self.ccl_crs, 0)
        device.clear_loaded_sub_device_manager()
        self.windows = 0

    def enter(self) -> int:
        """Load the split manager; returns the compute rows the window's owned ops must take as ``grid_rows``."""
        assert not self.active, "collective overlap windows do not nest"
        self.device.load_sub_device_manager(self.split)
        self.device.set_sub_device_stall_group(self.stall_group)
        self.active = True
        self.windows += 1
        return self.compute_rows

    def exit(self) -> None:
        assert self.active
        self.device.reset_sub_device_stall_group()
        self.device.clear_loaded_sub_device_manager()
        self.active = False

    def reduce_scatter(self, tensor, dim: int, cluster_axis: int):
        """The async reduce-scatter on the collective sub-device (inside a window)."""
        assert self.active, "reduce_scatter outside a collective overlap window"
        return ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            None,
            dim,
            self.semaphores,
            barrier_semaphore=self.barrier,
            num_links=self.num_links,
            topology=ttnn.Topology.Ring,
            subdevice_id=self.ccl_id,
            cluster_axis=cluster_axis,
            num_workers_per_link=self.workers_per_link,
        )
