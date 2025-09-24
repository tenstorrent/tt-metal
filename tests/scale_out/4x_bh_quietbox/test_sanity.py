import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn import ShardTensor2dMesh, ConcatMesh2dToTensor


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_DYNAMIC}], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param((4, 4), id="4x4_grid")], indirect=True)
def test_multihost_sanity(mesh_device):
    torch.manual_seed(0)

    shard_size = 32
    torch_tensor = torch.rand(
        (1, 1, shard_size * mesh_device.shape[0], shard_size * mesh_device.shape[1]), dtype=torch.bfloat16
    )
    torch_gelu = torch.nn.functional.gelu(torch_tensor)

    ttnn_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=(-2, -1)),
    )
    ttnn_tensor = ttnn.to_device(ttnn_tensor, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn_tensor = ttnn.gelu(ttnn_tensor)
    ttnn_loop_back_tensor = ttnn.from_device(ttnn_tensor)
    torch_loop_back_tensor = ttnn.to_torch(
        ttnn_loop_back_tensor,
        mesh_composer=ConcatMesh2dToTensor(mesh_device, mesh_shape=mesh_device.shape, dims=(-2, -1)),
    )

    assert_with_pcc(torch_gelu, torch_loop_back_tensor, pcc=0.9999)
