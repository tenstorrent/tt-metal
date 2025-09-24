# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((2, 4), id="2x4_grid"),
        pytest.param((1, 8), id="1x8_grid"),
    ],
    indirect=True,
)
def test_visualize_tensor_col_sharded(mesh_device):
    rows, cols = mesh_device.shape
    tile_size = 32
    full_tensor = torch.rand((1, 1, tile_size * rows, tile_size * cols), dtype=torch.bfloat16)
    mesh_mapper = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(
            placements=[
                ttnn.PlacementReplicate(),
                ttnn.PlacementShard(3),
            ],
        ),
    )
    ttnn_tensor = ttnn.from_torch(full_tensor, mesh_mapper=mesh_mapper, layout=ttnn.Layout.ROW_MAJOR)
    ttnn.visualize_tensor(ttnn_tensor)
    ttnn_tensor = ttnn_tensor.to(mesh_device)
    ttnn.visualize_tensor(ttnn_tensor)


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((2, 4), id="2x4_grid"),
        pytest.param((1, 8), id="1x8_grid"),
    ],
    indirect=True,
)
def test_visualize_tensor_row_sharded(mesh_device):
    rows, cols = mesh_device.shape
    tile_size = 32
    full_tensor = torch.rand((1, 1, tile_size * rows, tile_size * cols), dtype=torch.bfloat16)
    mesh_mapper = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(
            placements=[
                ttnn.PlacementShard(3),
                ttnn.PlacementReplicate(),
            ],
        ),
    )
    ttnn_tensor = ttnn.from_torch(full_tensor, mesh_mapper=mesh_mapper, layout=ttnn.Layout.ROW_MAJOR)
    ttnn.visualize_tensor(ttnn_tensor)
    ttnn_tensor = ttnn_tensor.to(mesh_device)
    ttnn.visualize_tensor(ttnn_tensor)


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4_grid")], indirect=True)
def test_visualize_tensor_2d_sharded(mesh_device):
    rows, cols = mesh_device.shape
    tile_size = 32
    full_tensor = torch.rand((1, 1, tile_size * rows, tile_size * cols), dtype=torch.bfloat16)
    mesh_mapper = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(
            placements=[
                ttnn.PlacementShard(2),
                ttnn.PlacementShard(3),
            ],
        ),
    )
    ttnn_tensor = ttnn.from_torch(full_tensor, mesh_mapper=mesh_mapper, layout=ttnn.Layout.ROW_MAJOR)
    ttnn.visualize_tensor(ttnn_tensor)
    ttnn_tensor = ttnn_tensor.to(mesh_device)
    ttnn.visualize_tensor(ttnn_tensor)


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4_grid")], indirect=True)
def test_visualize_tensor_row_major_distribution(mesh_device):
    rows, cols = mesh_device.shape
    tile_size = 32
    full_tensor = torch.rand((1, 1, tile_size * rows, tile_size * cols), dtype=torch.bfloat16)
    mesh_mapper = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementReplicate(), ttnn.PlacementShard(3)],
            mesh_shape_override=ttnn.MeshShape(1, 8),
        ),
    )
    ttnn_tensor = ttnn.from_torch(full_tensor, mesh_mapper=mesh_mapper, layout=ttnn.Layout.ROW_MAJOR)
    ttnn.visualize_tensor(ttnn_tensor)
    ttnn_tensor = ttnn_tensor.to(mesh_device)
    ttnn.visualize_tensor(ttnn_tensor)


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4_grid")], indirect=True)
def test_visualize_tensor_submesh_distribution(mesh_device):
    rows, cols = mesh_device.shape
    tile_size = 32
    full_tensor = torch.rand((1, 1, tile_size * rows, tile_size * cols), dtype=torch.bfloat16)
    mesh_mapper = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementReplicate(), ttnn.PlacementShard(3)],
            mesh_shape_override=ttnn.MeshShape(2, 2),
        ),
    )
    ttnn_tensor = ttnn.from_torch(full_tensor, mesh_mapper=mesh_mapper, layout=ttnn.Layout.ROW_MAJOR)
    ttnn.visualize_tensor(ttnn_tensor)
    ttnn_tensor = ttnn_tensor.to(mesh_device)
    ttnn.visualize_tensor(ttnn_tensor)


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4_grid")], indirect=True)
def test_visualize_tensor_1d_placements_and_override(mesh_device):
    rows, cols = mesh_device.shape
    tile_size = 32
    full_tensor = torch.rand((1, 1, tile_size * rows, tile_size * cols), dtype=torch.bfloat16)
    mesh_mapper = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementShard(3)],
            mesh_shape_override=ttnn.MeshShape([8]),
        ),
    )
    ttnn_tensor = ttnn.from_torch(full_tensor, mesh_mapper=mesh_mapper, layout=ttnn.Layout.ROW_MAJOR)
    ttnn.visualize_tensor(ttnn_tensor)
    ttnn_tensor = ttnn_tensor.to(mesh_device)
    ttnn.visualize_tensor(ttnn_tensor)
