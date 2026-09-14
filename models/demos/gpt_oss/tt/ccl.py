# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.modules.tt_ccl import get_num_links


class SamplingCCL:
    """Dedicated CCL handles for the TTSampling force-argmax all-gather.

    TTSampling's force-argmax path needs ``get_and_cycle_ag_semaphore_handles`` and a
    barrier accessor, neither of which CCLManager exposes.

    These handles are deliberately separate from CCLManager's ping-pong banks rather
    than borrowed from them. ``all_gather_async`` maps its two global semaphores to the
    forward and backward ring directions, so a handle that also serves the model's own
    CCLs as a single leaves per-direction counts behind and desyncs a later gather. On
    Blackhole that appears as argmax indices displaced by whole vocab chunks;
    ``models/demos/llama3_70b_galaxy/tt/llama_ccl.py`` records the same finding and
    solves it the same way. So these are fixed-role: one pair per axis, never cycled and
    never shared.

    Index 2 is the no-axis pool. ``TTSampling._get_sampling_cluster_axis`` returns None
    for a mesh whose shape contains a 1 (mesh_1x8) and the gather axis otherwise
    (mesh_2x4, mesh_4x4 and mesh_4x8 all give 1), so both forms reach these accessors.
    """

    _NO_AXIS = 2

    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        grid = mesh_device.compute_with_storage_grid_size()
        core_range_set = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
        )
        # all_gather_async consumes two semaphores per call; the barrier consumes one.
        self._ag_handles = [
            [ttnn.create_global_semaphore(mesh_device, core_range_set, 0) for _ in range(2)] for _ in range(3)
        ]
        self._barrier_handles = [ttnn.create_global_semaphore(mesh_device, core_range_set, 0) for _ in range(3)]

    def _pool_index(self, cluster_axis):
        return self._NO_AXIS if cluster_axis is None else cluster_axis

    def get_and_cycle_ag_semaphore_handles(self, cluster_axis=None):
        """Return the fixed pair for this axis. The name matches the interface TTSampling
        calls; the handles do not rotate, which is the point (see the class docstring)."""
        return self._ag_handles[self._pool_index(cluster_axis)]

    def get_sampling_barrier_semaphore_handle(self, cluster_axis=None):
        return self._barrier_handles[self._pool_index(cluster_axis)]

    def get_and_cycle_barrier_semaphore_handle(self, cluster_axis=None):
        # TTSampling reads the sampling accessor above through getattr with this method as
        # the default. Python evaluates that default eagerly, so it has to exist.
        return self.get_sampling_barrier_semaphore_handle(cluster_axis)

    def get_num_links(self, cluster_axis=None):
        return get_num_links(self.mesh_device, cluster_axis)

    def reset_global_semaphores(self):
        """Return every handle to 0, so a reused sampler starts from the state that trace
        capture assumes."""
        for pair in self._ag_handles:
            for semaphore in pair:
                ttnn.reset_global_semaphore_value(semaphore, 0)
        for semaphore in self._barrier_handles:
            ttnn.reset_global_semaphore_value(semaphore, 0)


class CCLManager:
    def __init__(self, mesh_device, num_links, topology=ttnn.Topology.Ring):
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology

        # Cache for ping pong buffers: key = (shape_tuple, dim, mesh_axis), value = [buffer1, buffer2]
        self._ping_pong_buffer_cache = {}
        self._ping_pong_buffer_indices = {}

        # Setup semaphores
        self._init_subdevice()

        # Initialize semaphores for reduce scatter and all gather
        self._init_semaphores()
        self.rs_ping_pong_idx = 0
        self.ag_ping_pong_idx = 0
        self.barrier_idx = 0

    def _init_subdevice(self):
        compute_grid_size = ttnn.CoreCoord(8, 8)
        self.ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
        )

        _worker_sub_device = ttnn.SubDevice(
            [
                self.ccl_cores,
            ]
        )
        self.ccl_sub_device_id = ttnn.SubDeviceId(0)

    def _init_semaphores(self):
        # Initialize semaphores for reduce scatter ping pong
        rs_n_sems = 3 * 2  # 3 semaphores * 2 for ping pong
        self.rs_ping_pong_semaphores = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(rs_n_sems)
        ]

        # Initialize semaphores for all gather ping pong
        ag_n_sems = 2 * 2  # 2 semaphores * 2 for ping pong (2 buffers)
        self.ag_ping_pong_semaphores = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(ag_n_sems)
        ]

        # Initialize barrier semaphores
        barrier_ns_sems = 2 * 1
        self.barrier_semaphore = [
            ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(barrier_ns_sems)
        ]

    def get_rs_ping_pong_semaphore(self):
        """
        Get semaphores for reduce scatter ping pong operations.

        Returns:
            List of 3 semaphores for the current ping pong cycle
        """
        cur_idx = self.rs_ping_pong_idx
        n_sems = 3
        self.rs_ping_pong_idx = (cur_idx + 1) % 2
        return self.rs_ping_pong_semaphores[cur_idx * n_sems : (cur_idx + 1) * n_sems]

    def get_ag_ping_pong_semaphore(self):
        """
        Get semaphores for all gather ping pong operations.

        Returns:
            List of 3 semaphores for the current ping pong cycle
        """
        cur_idx = self.ag_ping_pong_idx
        n_sems = 2
        self.ag_ping_pong_idx = (cur_idx + 1) % 2
        return self.ag_ping_pong_semaphores[cur_idx * n_sems : (cur_idx + 1) * n_sems]

    def get_barrier_semaphore(self):
        """
        Get semaphores for barrier operations.
        """
        cur_idx = self.barrier_idx
        self.barrier_idx = (cur_idx + 1) % 2
        return self.barrier_semaphore[cur_idx]

    def reset_global_semaphores(self):
        """Reset all global semaphores to 0"""
        for sem in self.rs_ping_pong_semaphores:
            ttnn.reset_global_semaphore_value(sem, 0)
        for sem in self.ag_ping_pong_semaphores:
            ttnn.reset_global_semaphore_value(sem, 0)
