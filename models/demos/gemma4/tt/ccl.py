# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
CCL manager for Gemma4 tensor parallelism.

Follows the TT_CCL pattern from tt_transformers/tt/ccl.py exactly:
- Per-axis semaphore sets (axis 0, axis 1, no-axis)
- Double-buffered (ping-pong) semaphores for each op type
- reduce_scatter: 3 semaphores per ping-pong set
- all_gather: 2 semaphores per ping-pong set
- barrier: 1 semaphore per ping-pong set
"""

import ttnn
from models.common.modules.tt_ccl import get_num_links as get_common_num_links


class CCLManager:
    def __init__(self, mesh_device, num_links, topology=ttnn.Topology.Linear):
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology

        compute_grid_size = mesh_device.compute_with_storage_grid_size()
        self.ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )
        self.ccl_sub_device_id = ttnn.SubDeviceId(0)

        # Per-axis semaphore arrays: index 0=axis-0, 1=axis-1, 2=no-axis
        self.barrier_semaphore_idx = [0, 0, 0]
        self.barrier_semaphore_handles = [[], [], []]
        self.ag_semaphores_idx = [0, 0, 0]
        self.ag_semaphore_handles = [[], [], []]
        self.rs_semaphores_idx = [0, 0, 0]
        self.rs_semaphore_handles = [[], [], []]

        for i in range(3):
            for _ in range(2):  # double-buffered
                self.barrier_semaphore_handles[i].append(ttnn.create_global_semaphore(mesh_device, self.ccl_cores, 0))
                self.ag_semaphore_handles[i].append(
                    [ttnn.create_global_semaphore(mesh_device, self.ccl_cores, 0) for _ in range(2)]
                )
                self.rs_semaphore_handles[i].append(
                    [ttnn.create_global_semaphore(mesh_device, self.ccl_cores, 0) for _ in range(3)]
                )

    def get_num_links(self, cluster_axis=None):
        return get_common_num_links(self.mesh_device, cluster_axis)

    def get_and_cycle_barrier_semaphore_handle(self, cluster_axis=None):
        idx = 2 if not cluster_axis else cluster_axis
        cur = self.barrier_semaphore_idx[idx]
        self.barrier_semaphore_idx[idx] = (cur + 1) % 2
        return self.barrier_semaphore_handles[idx][cur]

    def get_and_cycle_ag_semaphore_handles(self, cluster_axis=None):
        idx = 2 if not cluster_axis else cluster_axis
        cur = self.ag_semaphores_idx[idx]
        self.ag_semaphores_idx[idx] = (cur + 1) % 2
        return self.ag_semaphore_handles[idx][cur]

    def get_and_cycle_rs_semaphore_handles(self, cluster_axis=None):
        idx = 2 if not cluster_axis else cluster_axis
        cur = self.rs_semaphores_idx[idx]
        self.rs_semaphores_idx[idx] = (cur + 1) % 2
        return self.rs_semaphore_handles[idx][cur]
