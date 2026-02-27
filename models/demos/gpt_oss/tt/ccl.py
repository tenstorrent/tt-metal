# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import ttnn


class CCLManager:
    def __init__(self, mesh_device, mesh_shape, num_links=4, topology=ttnn.Topology.Ring, use_model_parallelism=False):
        self.mesh_device = mesh_device
        self.mesh_shape = mesh_shape
        self.num_links = num_links
        self.topology = topology
        self.use_model_parallelism = use_model_parallelism

        # Cache for ping pong buffers: key = (shape_tuple, dim, mesh_axis), value = [buffer1, buffer2]
        self._ping_pong_buffer_cache = {}
        self._ping_pong_buffer_indices = {}
        print("Use model parallelism:", self.use_model_parallelism)
        if self.use_model_parallelism:
            self._init_submeshes()
        else:
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

    def _init_submeshes(self):
        self.mp_submeshes = []
        for i in range(self.mesh_shape[1]):
            self.mp_submeshes.append(
                self.mesh_device.create_submesh(ttnn.MeshShape(self.mesh_shape[0], 1), ttnn.MeshCoordinate(0, i))
            )

        for x in self.mp_submeshes:
            print(f"Submesh shape={x.shape}, id={x.id()}")

        # Create socket pairs between submeshes for copying hidden_states in _forward_layers_and_head.
        # One pair per (from_id, to_id) with from_id != to_id; reused for all forward passes.
        num_submeshes = self.mesh_shape[1]
        self.submesh_socket_pairs = {}
        socket_memconfig = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 16 * 1024)
        for from_id in range(num_submeshes - 1):
            to_id = from_id + 1
            from_submesh = self.mp_submeshes[from_id]
            to_submesh = self.mp_submeshes[to_id]
            socket_connections = []
            for coord in ttnn.MeshCoordinateRange(from_submesh.shape):
                socket_connections.append(
                    ttnn.SocketConnection(
                        ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(0, 0)),
                        ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(0, 0)),
                    )
                )
            socket_config = ttnn.SocketConfig(socket_connections, socket_memconfig)
            sender_socket, receiver_socket = ttnn.create_socket_pair(from_submesh, to_submesh, socket_config)
            self.submesh_socket_pairs[(from_id, to_id)] = (sender_socket, receiver_socket)

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
