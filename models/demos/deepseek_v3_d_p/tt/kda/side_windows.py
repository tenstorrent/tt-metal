# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sub-device windows in which a KDA program that mostly waits on the SP fabric runs beside a projection.

The SP relay and the final carry each occupy a few worker cores and spend most of their time waiting for a peer. Run
under a sub-device manager that gives them the right-hand worker columns, the projection on the other columns runs at
the same time instead of after them:

* relay window: the relay on the right ``relay_columns`` columns, the output gate's projection on the others;
* final-carry window: the final carry on the last column, the output projection on the others.

Loading a manager drains the device and clearing it drains it again, so every program before a window has finished
when it starts, and every program in it before the next one starts. The switches are recorded in the trace. Outside a
trace no manager is loaded and the programs run one after another on the same cores.
"""

from __future__ import annotations

from contextlib import contextmanager

import ttnn


def _columns(first: int, last: int, rows: int) -> ttnn.CoreRangeSet:
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(first, 0), ttnn.CoreCoord(last, rows - 1))])


class SideWindows:
    """The two windows' sub-device managers on one mesh, created once and shared by its layers."""

    _shared: dict = {}

    @classmethod
    def shared(cls, mesh_device: ttnn.MeshDevice, relay_columns: int) -> SideWindows:
        key = (mesh_device.id(), relay_columns)
        if key not in cls._shared:
            cls._shared[key] = cls(mesh_device, relay_columns)
        return cls._shared[key]

    def __init__(self, mesh_device: ttnn.MeshDevice, relay_columns: int):
        grid = mesh_device.compute_with_storage_grid_size()
        if not 0 < relay_columns < grid.x:
            raise ValueError(f"the relay needs 1 to {grid.x - 1} worker columns, got {relay_columns}")
        self._mesh_device = mesh_device
        split = grid.x - relay_columns
        # Column-major, so the relay's head cores fill whole columns before its senders.
        self.relay_cores = [ttnn.CoreCoord(x, y) for x in range(split, grid.x) for y in range(grid.y)]
        # The output gate's and the output projection's worker blocks, from the origin.
        self.gate_grid = ttnn.CoreCoord(split, grid.y)
        self.projection_grid = ttnn.CoreCoord(grid.x - 1, grid.y)
        self._relay_manager = mesh_device.create_sub_device_manager(
            [ttnn.SubDevice([_columns(0, split - 1, grid.y)]), ttnn.SubDevice([_columns(split, grid.x - 1, grid.y)])],
            0,
        )
        self._carry_manager = mesh_device.create_sub_device_manager(
            [
                ttnn.SubDevice([_columns(0, grid.x - 2, grid.y)]),
                ttnn.SubDevice([_columns(grid.x - 1, grid.x - 1, grid.y)]),
            ],
            0,
        )

    @contextmanager
    def _window(self, manager):
        # Outside a trace the layer is bound by the host's time to queue it, which a switch adds to (about 45 us of
        # host time each) while the window saves only device time; the programs then run one after another.
        if not ttnn.device.is_trace_capture_active(self._mesh_device):
            yield
            return
        self._mesh_device.load_sub_device_manager(manager)
        try:
            yield
        finally:
            self._mesh_device.clear_loaded_sub_device_manager()

    def relay(self):
        """The relay on relay_cores, the output gate's projection on gate_grid."""
        return self._window(self._relay_manager)

    def final_carry(self):
        """The final carry on the last worker column, the output projection on projection_grid."""
        return self._window(self._carry_manager)
