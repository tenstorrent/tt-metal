# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class CCL:
    """
    A class to handle all CCL (Collective Communication Library) operations
    """

    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        self.grid = mesh_device.compute_with_storage_grid_size()
        self.num_cores = self.grid.x * self.grid.y
        self.core_range_set = ttnn.num_cores_to_corerangeset(self.num_cores, self.grid, row_wise=True)

        self.gather_sems = []
        self.reduce_scatter_sems = []
        self.barrier_sems = []
        for _ in range(len(list(mesh_device.shape))):
            self.gather_sems.append([])
            self.reduce_scatter_sems.append([])
            self.barrier_sems.append([])
            for _ in range(2):
                self.gather_sems[-1].append(
                    [
                        ttnn.create_global_semaphore(self.mesh_device, self.core_range_set, 0),
                        ttnn.create_global_semaphore(self.mesh_device, self.core_range_set, 0),
                    ]
                )  # use two semaphores to use minimal version of all_gather_async

                self.reduce_scatter_sems[-1].append(
                    [
                        ttnn.create_global_semaphore(self.mesh_device, self.core_range_set, 0),
                        ttnn.create_global_semaphore(self.mesh_device, self.core_range_set, 0),
                        ttnn.create_global_semaphore(self.mesh_device, self.core_range_set, 0),
                    ]
                )
                self.barrier_sems[-1].append(ttnn.create_global_semaphore(self.mesh_device, self.core_range_set, 0))

        # Synchronize the device to ensure that the semaphores are created
        ttnn.synchronize_device(self.mesh_device)

        # Each semaphore type needs its own independent counter
        self.gather_sem_cnt = [0, 0]
        self.reduce_scatter_sem_cnt = [0, 0]
        self.barrier_sem_cnt = [0, 0]

    def get_max_links(self, axis):
        """
        Get the maximum number of links for the given axis.
        """

        return 1  # Multi-link has PCC issues

        if axis == 0:
            return 4
        elif axis == 1:
            return 3
        else:
            raise ValueError("Axis must be 0 or 1.")

    def get_gather_sem(self, axis):
        """
        Get a semaphore for the given axis.
        """
        sem = self.gather_sems[axis][self.gather_sem_cnt[axis]]
        self.gather_sem_cnt[axis] = (self.gather_sem_cnt[axis] + 1) % 2
        return sem

    def get_buffer(self, key):
        """
        Get a buffer for the given key.
        """
        buffer = None

        return buffer

    def get_reduce_scatter_sem(self, axis):
        """
        Get a semaphore for the given axis.
        """
        sem = self.reduce_scatter_sems[axis][self.reduce_scatter_sem_cnt[axis]]
        self.reduce_scatter_sem_cnt[axis] = (self.reduce_scatter_sem_cnt[axis] + 1) % 2
        return sem

    def get_barrier_sem(self, axis):
        """
        Get a semaphore for the given axis.
        """
        sem = self.barrier_sems[axis][self.barrier_sem_cnt[axis]]
        self.barrier_sem_cnt[axis] = (self.barrier_sem_cnt[axis] + 1) % 2
        return sem

    def get_ccl_params_for_reduce_scatter(self, axis):
        """
        Get CCL parameters for reduce_scatter operations in execution order.
        This should be called at runtime in the forward pass.

        Args:
            axis: The cluster axis for the operation

        Returns:
            Dictionary containing semaphores and num_links for reduce_scatter
        """
        return {
            "multi_device_global_semaphore": self.get_reduce_scatter_sem(axis=axis),
            "barrier_semaphore": self.get_barrier_sem(axis=axis),
            "num_links": self.get_max_links(axis=axis),
        }

    def get_ccl_params_for_all_gather(self, axis):
        """
        Get CCL parameters for all_gather operations in execution order.
        This should be called at runtime in the forward pass.

        Args:
            axis: The cluster axis for the operation

        Returns:
            Dictionary containing semaphores and num_links for all_gather
        """
        return {
            "multi_device_global_semaphore": self.get_gather_sem(axis=axis),
            "barrier_semaphore": self.get_barrier_sem(axis=axis),
            "num_links": self.get_max_links(axis=axis),
        }

    def populate_all_gather_runtime_args(self, ccl_config: dict) -> dict:
        """Populate all_gather runtime arguments (semaphores, num_links) into the config.

        This method extracts the cluster_axis from the config, fetches the appropriate
        semaphores in execution order, and merges them with the static config.

        Args:
            ccl_config: Static CCL configuration dict (must contain 'cluster_axis')

        Returns:
            Complete configuration dict with both static and runtime parameters

        Example:
            ttnn.experimental.all_gather_async(x, **ccl.populate_all_gather_runtime_args(cfg["all_gather"]))
        """
        cluster_axis = ccl_config.get("cluster_axis")
        assert cluster_axis is not None, "cluster_axis must be present in CCL config"

        # Get runtime CCL parameters for all_gather
        ccl_params = self.get_ccl_params_for_all_gather(cluster_axis)

        # Merge static config with runtime CCL parameters
        return {**ccl_config, **ccl_params}

    def populate_reduce_scatter_runtime_args(self, ccl_config: dict) -> dict:
        """Populate reduce_scatter runtime arguments (semaphores, num_links) into the config.

        This method extracts the cluster_axis from the config, fetches the appropriate
        semaphores in execution order, and merges them with the static config.

        Args:
            ccl_config: Static CCL configuration dict (must contain 'cluster_axis')

        Returns:
            Complete configuration dict with both static and runtime parameters

        Example:
            ttnn.experimental.reduce_scatter_minimal_async(x, **ccl.populate_reduce_scatter_runtime_args(cfg["reduce_scatter"]))
        """
        cluster_axis = ccl_config.get("cluster_axis")
        assert cluster_axis is not None, "cluster_axis must be present in CCL config"

        # Get runtime CCL parameters for reduce_scatter
        ccl_params = self.get_ccl_params_for_reduce_scatter(cluster_axis)

        # Merge static config with runtime CCL parameters
        return {**ccl_config, **ccl_params}
