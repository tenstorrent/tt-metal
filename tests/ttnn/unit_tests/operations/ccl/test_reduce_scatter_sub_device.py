# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""reduce_scatter places its workers on the sub-device it is given, wherever that sub-device lies on the grid."""

import pytest
import torch
import ttnn


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_reduce_scatter_on_the_last_column(mesh_device):
    grid = mesh_device.compute_with_storage_grid_size()
    rest = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 2, grid.y - 1))})
    column = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(grid.x - 1, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([rest]), ttnn.SubDevice([column])], 0)
    mesh_device.load_sub_device_manager(manager)
    try:
        per_device = torch.randn(2, 1, 256, 1024)
        x = ttnn.from_torch(
            per_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        reduced = ttnn.reduce_scatter(
            x, 3, cluster_axis=1, subdevice_id=ttnn.SubDeviceId(1), topology=ttnn.Topology.Linear
        )
        got = ttnn.to_torch(reduced, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3)).float()
        expected = ttnn.to_torch(ttnn.from_torch(per_device, dtype=ttnn.bfloat16)).float().sum(dim=0, keepdim=True)
        error = ((got - expected).norm() / expected.norm()).item()
        assert error < 0.02, f"relative error {error}"
    finally:
        mesh_device.clear_loaded_sub_device_manager()
        mesh_device.remove_sub_device_manager(manager)
