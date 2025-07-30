# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import ttnn


class MetaLayerState:
    """
    A class to track the state of a meta layer.
    """

    def __init__(self, mesh_device: ttnn.MeshDevice, row: int = 0):
        self.mesh_device = mesh_device
        self.mesh_shape = list(mesh_device.shape)
        self.current_row = row

    def set_row(self, row: int) -> None:
        """
        Set the current state of the meta layer.
        """
        assert 0 <= row < self.mesh_shape[0], "Row index out of bounds"
        self.current_row = row

    def get_row(self) -> int:
        """
        Get the current state of the meta layer.
        """
        return self.current_row

    def get_current_device_coords(self) -> set[ttnn.MeshCoordinate]:
        """
        Get the devices in the current row.
        """
        device_coords = {(self.current_row, col) for col in range(self.mesh_shape[1])}
        return {ttnn.MeshCoordinate(*coord) for coord in device_coords}
