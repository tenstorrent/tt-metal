# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn


@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
@pytest.mark.parametrize("noc", [ttnn.NOC.NOC_0, ttnn.NOC.NOC_1])
def test_dram_assignment_at_mesh_coordinate(mesh_device, noc, expect_error):
    """The query returns one valid, ordered worker per bank on every local die."""
    query = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment_at_mesh_coordinate
    reference = ttnn.get_optimal_dram_bank_to_logical_worker_assignment(mesh_device, noc)
    for col in range(4):
        assignment = query(mesh_device, noc, ttnn.MeshCoordinate(0, col))
        grid = mesh_device.compute_with_storage_grid_size()
        assert len(assignment) == len(reference)
        assert len(set((core.x, core.y) for core in assignment)) == len(assignment)
        assert all(0 <= core.x < grid.x and 0 <= core.y < grid.y for core in assignment)
        unit_mesh = mesh_device.create_submesh(ttnn.MeshShape(1, 1), offset=ttnn.MeshCoordinate(0, col))
        assert assignment == ttnn.get_optimal_dram_bank_to_logical_worker_assignment(unit_mesh, noc)
    with expect_error(RuntimeError, "Coordinate"):
        query(mesh_device, noc, ttnn.MeshCoordinate(0, 4))
