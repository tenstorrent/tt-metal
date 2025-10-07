# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn


class CCL:
    """
    A class to handle all CCL (Collective Communication Library) operations
    with separate semaphore management for prefill and decode phases.
    """

    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        self.grid = mesh_device.compute_with_storage_grid_size()
        self.num_cores = self.grid.x * self.grid.y
        self.core_range_set = ttnn.num_cores_to_corerangeset(self.num_cores, self.grid, row_wise=True)

        # Two top-level phases
        self.phases = ["prefill", "decode"]

        # Current phase (set by model config methods)
        self.current_phase = None

        # Dictionary to hold semaphores per phase
        self.sems = {
            phase: {"gather": [], "from": [], "to": [], "reduce_scatter": [], "barrier": []} for phase in self.phases
        }

        # Separate sem_cnt tracking per phase
        self.sem_cnt = {phase: [0, 0] for phase in self.phases}

        # Initialize semaphores
        for phase in self.phases:
            for _ in range(len(list(mesh_device.shape))):
                self.sems[phase]["gather"].append([])
                self.sems[phase]["from"].append([])
                self.sems[phase]["to"].append([])
                self.sems[phase]["reduce_scatter"].append([])
                self.sems[phase]["barrier"].append([])

                for _ in range(2):  # Double buffered
                    # Gather semaphores (2 per axis)
                    self.sems[phase]["gather"][-1].append(
                        [
                            ttnn.create_global_semaphore(self.mesh_device, self.core_range_set, 0),
                            ttnn.create_global_semaphore(self.mesh_device, self.core_range_set, 0),
                        ]
                    )
                    # From/to semaphores
                    self.sems[phase]["from"][-1].append(
                        ttnn.create_global_semaphore(self.mesh_device, self.core_range_set, 0)
                    )
                    self.sems[phase]["to"][-1].append(
                        ttnn.create_global_semaphore(self.mesh_device, self.core_range_set, 0)
                    )
                    # Reduce scatter (3 per axis)
                    self.sems[phase]["reduce_scatter"][-1].append(
                        [
                            ttnn.create_global_semaphore(self.mesh_device, self.core_range_set, 0),
                            ttnn.create_global_semaphore(self.mesh_device, self.core_range_set, 0),
                            ttnn.create_global_semaphore(self.mesh_device, self.core_range_set, 0),
                        ]
                    )
                    # Barrier
                    self.sems[phase]["barrier"][-1].append(
                        ttnn.create_global_semaphore(self.mesh_device, self.core_range_set, 0)
                    )

    def set_phase(self, phase: str):
        """Set the current phase for subsequent operations."""
        if phase not in self.phases:
            raise ValueError(f"Invalid phase: {phase}. Must be one of {self.phases}")
        self.current_phase = phase

    def get_max_links(self, axis: int):
        """Get the maximum number of links for the given axis."""
        return 1  # Multi-link has PCC issues
        # Uncomment if multi-link is supported:
        # if axis == 0:
        #     return 4
        # elif axis == 1:
        #     return 3
        # else:
        #     raise ValueError("Axis must be 0 or 1.")

    def _get_sem(self, sem_type: str, axis: int):
        """Helper: get a semaphore for the current phase, type, and axis."""
        if self.current_phase is None:
            raise ValueError("Phase not set. Call set_phase() first.")

        phase = self.current_phase

        if sem_type not in self.sems[phase]:
            raise ValueError(f"Unknown sem_type: {sem_type}")

        if axis not in range(len(self.sems[phase][sem_type])):
            raise ValueError(f"Unknown axis: {axis}")

        sem = self.sems[phase][sem_type][axis][self.sem_cnt[phase][axis]]
        self.sem_cnt[phase][axis] = (self.sem_cnt[phase][axis] + 1) % 2
        return sem

    def get_gather_sem(self, axis: int):
        return self._get_sem("gather", axis)

    def get_from_sem(self, axis: int):
        return self._get_sem("from", axis)

    def get_to_sem(self, axis: int):
        return self._get_sem("to", axis)

    def get_reduce_scatter_sem(self, axis: int):
        return self._get_sem("reduce_scatter", axis)

    def get_barrier_sem(self, axis: int):
        return self._get_sem("barrier", axis)

    def get_reduce_scatter_params(self, axis: int):
        """Get parameters for reduce scatter operations."""
        return {
            "multi_device_global_semaphore": self.get_reduce_scatter_sem(axis=axis),
            "barrier_semaphore": self.get_barrier_sem(axis=axis),
            "num_links": self.get_max_links(axis=axis),
        }

    def get_all_gather_params(self, axis: int):
        """Get parameters for all gather operations."""
        return {
            "multi_device_global_semaphore": self.get_gather_sem(axis=axis),
            "barrier_semaphore": self.get_barrier_sem(axis=axis),
            "num_links": self.get_max_links(axis=axis),
        }

    def clear_phase(self, phase: str):
        """
        Delete all semaphores for a given phase (e.g., after prefill).
        """
        if phase not in self.sems:
            raise ValueError(f"Unknown phase: {phase}")
        self.sems[phase] = {"gather": [], "from": [], "to": [], "reduce_scatter": [], "barrier": []}
        self.sem_cnt[phase] = [0, 0]
