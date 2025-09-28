# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from ..utils.tensor import bf16_tensor


class CCLManager:
    """
    Manages parallelization of DiT model.

        - stores mesh device, num links, topology
        - caches ping pong buffers and semaphores
        - sets up one SubDevice spanning all compute cores
    """

    def __init__(
        self,
        mesh_device,
        num_links=1,
        topology=None,
    ):
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
        self.rs_ping_pong_idx = [0, 0]
        self.ag_ping_pong_idx = [0, 0]
        self.barrier_idx = [0, 0]

    def _init_subdevice(self):
        compute_grid_size = self.mesh_device.compute_with_storage_grid_size()
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
        # Initialize semaphores for reduce scatter ping pong - separate for each mesh axis
        rs_n_sems = 3 * 2  # 3 semaphores * 2 for ping pong
        self.rs_ping_pong_semaphores = {
            0: [ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(rs_n_sems)],
            1: [ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(rs_n_sems)],
        }

        # Initialize semaphores for all gather ping pong - separate for each mesh axis
        ag_n_sems = 2 * 2  # 2 semaphores * 2 for ping pong (2 buffers)
        self.ag_ping_pong_semaphores = {
            0: [ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(ag_n_sems)],
            1: [ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(ag_n_sems)],
        }

        # Initialize barrier semaphore
        barrier_n_sems = 1 * 2
        self.barrier_semaphores = {
            0: [ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(barrier_n_sems)],
            1: [ttnn.create_global_semaphore(self.mesh_device, self.ccl_cores, 0) for _ in range(barrier_n_sems)],
        }

    def get_rs_ping_pong_buffer(self, shape, dim, mesh_axis):
        """
        Get or create ping pong buffers for reduce scatter operations.
        Caches buffers based on shape, dim, and mesh_axis.

        Args:
            shape: Tensor shape tuple
            dim: Dimension for the operation
            mesh_axis: Mesh axis for parallelization

        Returns:
            Current ping pong buffer (alternates between two buffers)
        """
        # Create cache key from the parameters
        cache_key = (tuple(shape), dim, mesh_axis)

        # Create buffers if not cached
        if cache_key not in self._ping_pong_buffer_cache:
            # Synchronize devices to ensure all are ready to allocate and proceed
            ttnn.synchronize_device(self.mesh_device)
            # Create two buffers for ping pong
            buffers = []
            output_buffer_shape = list(shape)
            output_buffer_shape[dim] //= self.mesh_device.shape[mesh_axis]

            intermediate_buffer_shape = list(shape)
            intermediate_buffer_shape = [2] + intermediate_buffer_shape
            for _ in range(2):
                intermediate_buffer = bf16_tensor(torch.empty(intermediate_buffer_shape), device=self.mesh_device)
                output_buffer = bf16_tensor(torch.empty(output_buffer_shape), device=self.mesh_device)
                buffers.append([intermediate_buffer, output_buffer])

            self._ping_pong_buffer_cache[cache_key] = buffers
            self._ping_pong_buffer_indices[cache_key] = 0

        # Get current buffer and alternate index
        current_idx = self._ping_pong_buffer_indices[cache_key]
        self._ping_pong_buffer_indices[cache_key] = 1 - current_idx

        return self._ping_pong_buffer_cache[cache_key][current_idx]

    def get_ag_ping_pong_buffer(self, shape, dim, mesh_axis):
        """
        Get or create ping pong buffers for all gather operations.
        Caches buffers based on shape, dim, and mesh_axis.

        Args:
            shape: Tensor shape tuple
            dim: Dimension for the operation
            mesh_axis: Mesh axis for parallelization

        Returns:
            Current ping pong buffer (alternates between two buffers)
        """
        # Create cache key from the parameters (use different namespace than rs)
        cache_key = ("ag", tuple(shape), dim, mesh_axis)

        # Create buffers if not cached
        if cache_key not in self._ping_pong_buffer_cache:
            # Synchronize devices to ensure all are ready to allocate and proceed
            ttnn.synchronize_device(self.mesh_device)
            # Create two buffers for ping pong
            buffers = []
            output_buffer_shape = list(shape)
            output_buffer_shape[dim] *= self.mesh_device.shape[mesh_axis]  # All gather increases size
            for _ in range(2):
                output_buffer = bf16_tensor(torch.empty(output_buffer_shape), device=self.mesh_device)
                buffers.append(output_buffer)

            self._ping_pong_buffer_cache[cache_key] = buffers
            self._ping_pong_buffer_indices[cache_key] = 0

        # Get current buffer and alternate index
        current_idx = self._ping_pong_buffer_indices[cache_key]
        self._ping_pong_buffer_indices[cache_key] = 1 - current_idx

        return self._ping_pong_buffer_cache[cache_key][current_idx]

    def get_rs_ping_pong_semaphore(self, mesh_axis):
        """
        Get semaphores for reduce scatter ping pong operations.

        Args:
            mesh_axis: The mesh axis (0 or 1) to get semaphores for

        Returns:
            List of 3 semaphores for the current ping pong cycle
        """
        cur_idx = self.rs_ping_pong_idx[mesh_axis]
        n_sems = 3
        self.rs_ping_pong_idx[mesh_axis] = (cur_idx + 1) % 2
        return self.rs_ping_pong_semaphores[mesh_axis][cur_idx * n_sems : (cur_idx + 1) * n_sems]

    def get_ag_ping_pong_semaphore(self, mesh_axis):
        """
        Get semaphores for all gather ping pong operations.

        Args:
            mesh_axis: The mesh axis (0 or 1) to get semaphores for

        Returns:
            List of 2 semaphores for the current ping pong cycle
        """
        cur_idx = self.ag_ping_pong_idx[mesh_axis]
        n_sems = 2
        self.ag_ping_pong_idx[mesh_axis] = (cur_idx + 1) % 2
        return self.ag_ping_pong_semaphores[mesh_axis][cur_idx * n_sems : (cur_idx + 1) * n_sems]

    def get_barrier_semaphore(self, mesh_axis):
        """
        Get semaphore for barrier operations.
        """
        cur_idx = self.barrier_idx[mesh_axis]
        n_sems = 1
        self.barrier_idx[mesh_axis] = (cur_idx + 1) % 2
        return self.barrier_semaphores[mesh_axis][cur_idx]

    def reset_global_semaphores(self):
        """Reset all global semaphores to 0"""
        for axis in [0, 1]:
            for sem in self.rs_ping_pong_semaphores[axis]:
                ttnn.reset_global_semaphore_value(sem, 0)
            for sem in self.ag_ping_pong_semaphores[axis]:
                ttnn.reset_global_semaphore_value(sem, 0)

    def get_ag_hyperparams(self, shape):
        if shape[2] > 512:
            return {
                "chunks_per_sync": 16,
                "num_workers_per_link": 3,
                "num_buffers_per_channel": 2,
            }
        else:
            return {
                "chunks_per_sync": 10,
                "num_workers_per_link": 2,
                "num_buffers_per_channel": 2,
            }

    def get_rs_hyperparams(self, shape):
        return {
            "chunks_per_sync": 2,
            "num_workers_per_link": 2,
            "num_buffers_per_channel": 2,
        }
