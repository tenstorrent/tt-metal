# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn


class CCLManager:
    def __init__(self, mesh_device, num_links, topology=ttnn.Topology.Ring):
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology

        # Cache for ping pong buffers: key = (shape_tuple, dim, mesh_axis), value = [buffer1, buffer2]
        self._ping_pong_buffer_cache = {}
        self._ping_pong_buffer_indices = {}
        self._decode_all_reduce = {}
        self._decode_gate_buffers = {}

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

    def get_decode_all_reduce(self, hidden_size, cluster_axis):
        """Persistent fused all-reduce shared by every decoder layer's decode path (see fused_decode.py)."""
        from .fused_decode import DecodeAllReduce

        key = (hidden_size, cluster_axis)
        if key not in self._decode_all_reduce:
            self._decode_all_reduce[key] = DecodeAllReduce(self.mesh_device, hidden_size, cluster_axis, self.topology)
        return self._decode_all_reduce[key]

    def get_decode_gate_buffers(self, num_experts, width, tokens):
        """Router-gate constants and output buffers shared by every decoder layer's decode path."""
        from .fused_decode import create_decode_gate_buffers

        key = (num_experts, width, tokens)
        if key not in self._decode_gate_buffers:
            self._decode_gate_buffers[key] = create_decode_gate_buffers(self.mesh_device, num_experts, width, tokens)
        return self._decode_gate_buffers[key]

    def reset_global_semaphores(self):
        """Reset all global semaphores to 0"""
        for sem in self.rs_ping_pong_semaphores:
            ttnn.reset_global_semaphore_value(sem, 0)
        for sem in self.ag_ping_pong_semaphores:
            ttnn.reset_global_semaphore_value(sem, 0)
