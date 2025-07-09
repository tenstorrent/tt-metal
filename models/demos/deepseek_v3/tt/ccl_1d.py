# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class CCL1D:
    """
    A class to handle all CCL (Collective Communication Library) operations
    """

    def __init__(self, hf_config, mesh_device):
        self.mesh_device = mesh_device
        self.grid = mesh_device.compute_with_storage_grid_size()
        self.num_cores = self.grid.x * self.grid.y
        self.core_range_set = ttnn.num_cores_to_corerangeset(self.num_cores, self.grid, row_wise=True)

        self.sems = []
        for _ in range(len(list(mesh_device.shape))):
            self.sems.append([])
            for _ in range(2):
                self.sems[-1].append(ttnn.create_global_semaphore(self.mesh_device, self.core_range_set, 0))
        self.sem_cnt = [0, 0]

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

    def get_semaphore(self, axis):
        """
        Get a semaphore for the given axis.
        """
        sem = self.sems[axis][self.sem_cnt[axis]]
        self.sem_cnt[axis] = (self.sem_cnt[axis] + 1) % 2
        return sem

    def get_buffer(self, key):
        """
        Get a buffer for the given key.
        """
        buffer = None

        return buffer
