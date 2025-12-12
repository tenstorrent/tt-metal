import pytest
import torch
import ttnn
from models.common.utility_functions import comp_pcc, comp_allclose


@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
def test_matmul(mesh_device):
    w_in = torch.load("models/demos/llama3_70b_galaxy/tests/w1_weight.pt")
    x_in = torch.load("models/demos/llama3_70b_galaxy/tests/w1_in.pt")
    ref_after_w1 = torch.load("models/demos/llama3_70b_galaxy/tests/ref_after_w1.pt")
    comp_out = torch.load("models/demos/llama3_70b_galaxy/tests/comp_out.pt")

    w_in = ttnn.from_torch(
        w_in,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=[8, 4]),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    x_in = ttnn.from_torch(
        x_in,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=[8, 4]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    out = ttnn.matmul(x_in, w_in, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG)
    mesh_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 0), mesh_shape=[8, 4])
    out = ttnn.to_torch(out, mesh_composer=mesh_composer).sum(0)
    out = torch.permute(out, (1, 0, 2))
    passing, pcc_message = comp_pcc(ref_after_w1, out)
    print(f"Non-Prefetch Matmul PCC with reference: {pcc_message}")
    print(comp_allclose(ref_after_w1, out))

    passing, pcc_message = comp_pcc(comp_out, out)
    print(f"Non-Prefetch Matmul PCC with ring matmul output: {pcc_message}")
    print(comp_allclose(comp_out, out))
