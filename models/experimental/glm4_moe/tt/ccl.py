# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

"""CCL (Collective Communication Library) semaphore management for GLM-4.7-REAP.

Copied from models/demos/deepseek_v3/tt/ccl.py with no modifications.
Manages global semaphores for async reduce_scatter and all_gather operations
across a 2D mesh device (TG Galaxy: Mesh(8,4), TP=axis 0, DP=axis 1).

Usage:
    ccl = CCL(mesh_device)
    # Before trace capture:  ccl.reset_sem_counters()
    # Before trace replay:   ccl.reset_sem_counters()
    # After each prefill:    ccl.reset_sem_counters()
    # After each decode:     ccl.reset_sem_counters()
"""

import os

import ttnn


def glm4_moe_ccl_num_links_for_axis(axis: int) -> int:
    """Number of fabric links for async CCL ops (reduce_scatter / all_gather).

    Env (in order of precedence):
      - ``GLM4_MOE_CCL_NUM_LINKS_AXIS{axis}`` (0 or 1 for TG mesh axes)
      - ``GLM4_MOE_CCL_NUM_LINKS`` (applies to all axes if per-axis unset)

    Default ``1`` matches prior behavior (multi-link was disabled due to PCC risk).
    """
    ax_raw = os.environ.get(f"GLM4_MOE_CCL_NUM_LINKS_AXIS{int(axis)}", "").strip()
    if ax_raw:
        return max(1, int(ax_raw))
    glob = os.environ.get("GLM4_MOE_CCL_NUM_LINKS", "").strip()
    if glob:
        return max(1, int(glob))
    return 1


def glm4_moe_ccl_topology_for_collectives() -> ttnn.Topology:
    """Topology for experimental async gather/scatter and simple all_gather helpers.

    Env: ``GLM4_MOE_CCL_TOPOLOGY`` = ``linear`` (default) or ``ring`` / ``1d_ring``.
    """
    t = os.environ.get("GLM4_MOE_CCL_TOPOLOGY", "linear").strip().lower()
    if t in ("ring", "1d_ring"):
        return ttnn.Topology.Ring
    return ttnn.Topology.Linear


class CCL:
    """
    A class to handle all CCL (Collective Communication Library) operations
    """

    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        self.grid = mesh_device.compute_with_storage_grid_size()
        self.num_cores = self.grid.x * self.grid.y
        self.core_range_set = ttnn.num_cores_to_corerangeset(self.num_cores, self.grid, row_wise=True)
        self.num_axes = len(list(self.mesh_device.shape))
        self.sems_per_axis = 2

        self.gather_sems = []
        self.reduce_scatter_sems = []
        self.barrier_sems = []
        for _ in range(len(list(mesh_device.shape))):
            self.gather_sems.append([])
            self.reduce_scatter_sems.append([])
            self.barrier_sems.append([])
            for _ in range(self.sems_per_axis):
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

        # Each semaphore type needs its own independent counter for each cluster axis
        self.gather_sem_cnt = [0 for _ in range(self.num_axes)]
        self.reduce_scatter_sem_cnt = [0 for _ in range(self.num_axes)]
        self.barrier_sem_cnt = [0 for _ in range(self.num_axes)]

    def get_max_links(self, axis):
        """Return num_links for CCL async ops; see ``glm4_moe_ccl_num_links_for_axis``."""
        return glm4_moe_ccl_num_links_for_axis(int(axis))

    def _get_sem_and_update_counter(self, sem_list, counter_list, axis):
        """
        Helper method to get a semaphore and update its counter.

        Args:
            sem_list: The semaphore list to select from
            counter_list: The counter list to update
            axis: The cluster axis

        Returns:
            The selected semaphore
        """
        sem = sem_list[axis][counter_list[axis]]
        counter_list[axis] = (counter_list[axis] + 1) % self.sems_per_axis
        return sem

    def get_gather_sem(self, axis):
        """
        Get a semaphore for the given axis.
        """
        return self._get_sem_and_update_counter(self.gather_sems, self.gather_sem_cnt, axis)

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
        return self._get_sem_and_update_counter(self.reduce_scatter_sems, self.reduce_scatter_sem_cnt, axis)

    def get_barrier_sem(self, axis):
        """
        Get a semaphore for the given axis.
        """
        return self._get_sem_and_update_counter(self.barrier_sems, self.barrier_sem_cnt, axis)

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

    def reset_sem_counters(self):
        """Reset the semaphore counters for all axes."""
        self.gather_sem_cnt = [0 for _ in range(self.num_axes)]
        self.reduce_scatter_sem_cnt = [0 for _ in range(self.num_axes)]
        self.barrier_sem_cnt = [0 for _ in range(self.num_axes)]
