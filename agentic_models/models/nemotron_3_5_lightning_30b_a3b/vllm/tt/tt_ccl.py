# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""`models.common.tt_ccl.TT_CCL` with the bring-up's lazy (`eager=False`) semaphore allocation.

Vendored from the Nemotron bring-up's edit of `models/common/tt_ccl.py` (tt-metal 62cee5d6) as a subclass of the
pinned class, so the shared module stays unchanged. The sampler only needs the flattened-mesh barrier and
all-gather semaphores; allocating every axis and family eagerly takes L1 the grouped MoE kernel needs.
"""

import ttnn
from models.common.tt_ccl import TT_CCL as _TT_CCL


class TT_CCL(_TT_CCL):
    def __init__(self, mesh_device, *, eager: bool = True):
        """Own double-buffered CCL semaphores.

        With ``eager=False``, each accessor allocates both slots of its axis/family on first use. Call every
        required accessor during warmup, before trace capture; subsequent calls only cycle existing handles.
        ``eager=True`` is the pinned class unchanged.
        """
        if eager:
            super().__init__(mesh_device)
            return
        self.mesh_device = mesh_device
        grid = mesh_device.compute_with_storage_grid_size()
        self.sub_device_crs = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
        )
        self.barrier_semaphore_idx = [0, 0, 0]
        self.barrier_semaphore_handles = [[], [], []]
        self.ag_semaphores_idx = [0, 0, 0]
        self.ag_semaphore_handles = [[], [], []]
        self.rs_semaphores_idx = [0, 0, 0]
        self.rs_semaphore_handles = [[], [], []]

    def _semaphore(self):
        return ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)

    def get_and_cycle_barrier_semaphore_handle(self, cluster_axis=None):
        index = 2 if cluster_axis is None else cluster_axis
        if not self.barrier_semaphore_handles[index]:
            self.barrier_semaphore_handles[index] = [self._semaphore() for _ in range(2)]
        return super().get_and_cycle_barrier_semaphore_handle(cluster_axis)

    def get_and_cycle_ag_semaphore_handles(self, cluster_axis=None):
        index = 2 if cluster_axis is None else cluster_axis
        if not self.ag_semaphore_handles[index]:
            self.ag_semaphore_handles[index] = [[self._semaphore() for _ in range(2)] for _ in range(2)]
        return super().get_and_cycle_ag_semaphore_handles(cluster_axis)

    def get_and_cycle_rs_semaphore_handles(self, cluster_axis=None):
        index = 2 if cluster_axis is None else cluster_axis
        if not self.rs_semaphore_handles[index]:
            self.rs_semaphore_handles[index] = [[self._semaphore() for _ in range(3)] for _ in range(2)]
        return super().get_and_cycle_rs_semaphore_handles(cluster_axis)
