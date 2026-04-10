# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn


class CCLManager:
    """CCL manager for Gemma4 tensor parallelism.

    Creates global semaphores needed for async reduce_scatter + all_gather
    (the decomposed all-reduce used on T3K and larger meshes).

    On N300 (TP=2), the simpler ttnn.all_reduce works fine.
    On T3K (TP=8), we must use the async reduce_scatter + all_gather decomposition.
    """

    def __init__(self, mesh_device, num_links=1, topology=ttnn.Topology.Linear):
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology
        self.num_devices = mesh_device.get_num_devices()

        # Create semaphores for async CCL ops (needed for T3K)
        grid = mesh_device.compute_with_storage_grid_size()
        num_cores = grid.x * grid.y
        core_range_set = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)

        # Double-buffered semaphore slots, matching tt_transformers TT_CCL layout:
        #   rs: each slot is a list of 3 semaphores
        #   ag: each slot is a list of 2 semaphores
        #   barrier: each slot is a single semaphore
        self._rs_semaphores = []
        self._ag_semaphores = []
        self._barrier_semaphores = []
        for _ in range(2):  # double-buffered
            self._rs_semaphores.append([ttnn.create_global_semaphore(mesh_device, core_range_set, 0) for _ in range(3)])
            self._ag_semaphores.append([ttnn.create_global_semaphore(mesh_device, core_range_set, 0) for _ in range(2)])
            self._barrier_semaphores.append(ttnn.create_global_semaphore(mesh_device, core_range_set, 0))
        ttnn.synchronize_device(mesh_device)

        # Rotating counters for double-buffering
        self._rs_idx = 0
        self._ag_idx = 0
        self._barrier_idx = 0

    def get_rs_semaphore(self):
        """Returns list of 3 semaphores for reduce_scatter (cycles double-buffer)."""
        sems = self._rs_semaphores[self._rs_idx]
        self._rs_idx = (self._rs_idx + 1) % 2
        return sems

    def get_ag_semaphore(self):
        """Returns list of 2 semaphores for all_gather (cycles double-buffer)."""
        sems = self._ag_semaphores[self._ag_idx]
        self._ag_idx = (self._ag_idx + 1) % 2
        return sems

    def get_barrier_semaphore(self):
        """Returns single barrier semaphore (cycles double-buffer)."""
        sem = self._barrier_semaphores[self._barrier_idx]
        self._barrier_idx = (self._barrier_idx + 1) % 2
        return sem


def ccl_allreduce(tensor, mesh_config, ccl_manager, memory_config=None):
    """All-reduce that works on both N300 (TP=2) and T3K (TP=8).

    N300 (TP<=2): uses simple ttnn.all_reduce.
    T3K (TP>2):  uses reduce_scatter + all_gather with async semaphores.
    """
    if mesh_config is None or mesh_config.tp <= 1:
        return tensor

    memory_config = memory_config or ttnn.DRAM_MEMORY_CONFIG
    tp_axis = mesh_config.tp_axis

    if ccl_manager is not None and ccl_manager.num_devices > 2:
        # T3K path: decompose into reduce_scatter + all_gather
        scattered = ttnn.experimental.reduce_scatter_minimal_async(
            tensor,
            dim=3,
            cluster_axis=tp_axis,
            num_links=ccl_manager.num_links,
            topology=ccl_manager.topology,
            multi_device_global_semaphore=ccl_manager.get_rs_semaphore(),
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
            memory_config=memory_config,
        )
        tensor.deallocate(True)
        gathered = ttnn.experimental.all_gather_async(
            scattered,
            dim=3,
            cluster_axis=tp_axis,
            mesh_device=ccl_manager.mesh_device,
            num_links=ccl_manager.num_links,
            topology=ccl_manager.topology,
            multi_device_global_semaphore=ccl_manager.get_ag_semaphore(),
            barrier_semaphore=ccl_manager.get_barrier_semaphore(),
            memory_config=memory_config,
        )
        scattered.deallocate(True)
        return gathered
    else:
        # N300 path: simple all_reduce
        result = ttnn.all_reduce(
            tensor,
            cluster_axis=tp_axis,
            num_links=1,
            topology=ttnn.Topology.Linear,
            memory_config=memory_config,
        )
        tensor.deallocate(True)
        return result
