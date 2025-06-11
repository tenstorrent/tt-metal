from math import prod
from time import sleep


import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


TEST_SHAPES = [
    (1, 1, 2, 16),
    (1, 1, 1, 16),
    (1, 1, 1, 64),
    (1, 1, 3, 128),
    (1, 13, 1, 32),
    (1, 1, 1, 512),
    (100, 1, 1, 16),
    (1, 1, 1, 17),
]


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("coords", [((0, 0), (0, 1))])
@pytest.mark.parametrize("shape", TEST_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_send_receive(mesh_device, coords, shape, layout, dtype):
    use_shape = tuple(s * (2 if i == 0 else 1) for i, s in enumerate(shape))

    coord0, coord1 = (ttnn.MeshCoordinate(c) for c in coords)

    input_tensor_torch = torch.linspace(1, prod(use_shape), prod(use_shape)).reshape(use_shape).to(dtype=dtype)
    input_tensor = ttnn.from_torch(
        input_tensor_torch, layout=layout, device=mesh_device, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0)
    )

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    send_semaphore = ttnn.create_global_semaphore(mesh_device, cores, 0)
    receiver_semaphore = ttnn.create_global_semaphore(mesh_device, cores, 0)

    sent_tensor = ttnn.point_to_point(
        input_tensor,
        coord0,
        coord1,
        ttnn.Topology.Linear,
        receiver_semaphore,
    )
    torch_sent_tensor = ttnn.to_torch(sent_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    return_tensor = ttnn.point_to_point(sent_tensor, coord1, coord0, ttnn.Topology.Linear, send_semaphore)

    torch_return_tensor = ttnn.to_torch(return_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    assert_with_pcc(torch_return_tensor[:1, :, :, :], input_tensor_torch[:1, :, :, :])
