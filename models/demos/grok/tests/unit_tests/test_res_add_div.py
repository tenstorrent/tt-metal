import math

import torch

import ttnn


def test_res_add_div(mesh_device):
    torch_a = torch.randn(1, 1, 32, 8192)
    torch_b = torch.randn(1, 1, 32, 8192)

    ref_output = (torch_a + torch_b) / math.sqrt(2)

    tt_a = ttnn.from_torch(
        torch_a,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=(8, 4)),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_b = ttnn.from_torch(
        torch_b,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=(8, 4)),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = ttnn.add(tt_a, tt_b)
    tt_output = ttnn.div(tt_output, math.sqrt(2))

    breakpoint()
