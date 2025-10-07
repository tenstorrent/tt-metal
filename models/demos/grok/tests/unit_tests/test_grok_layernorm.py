import pytest
import torch

import ttnn
from models.demos.grok.tt.ccl import CCL_Manager, tt_distributed_rmsnorm


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
def test_grok_layernorm(mesh_device):
    tt_ccl = CCL_Manager(mesh_device)

    torch_gamma = torch.randn(8192)

    torch_input = torch.randn(1, 1, 32, 8192)
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 3), mesh_shape=(8, 4)),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_gamma = ttnn.from_torch(
        torch_gamma.unsqueeze(0).unsqueeze(0).unsqueeze(0).reshape(1, 1, 8192 // 32, 32),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=(8, 4)),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_layernorm = torch.nn.LayerNorm(torch_input.shape[-1], eps=1e-5)
    torch_layernorm.weight = torch.nn.Parameter(torch_gamma)
    torch_layernorm_output = torch_layernorm(torch_input)

    tt_output = tt_distributed_rmsnorm(
        tt_input,
        1e-5,
        tt_gamma,
        mesh_device,
        tt_ccl,
        ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False
        ),
    )

    tt_output_torch = ttnn.to_torch(
        tt_output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=(8, 4))
    )[:1, :, :, :]
    breakpoint()
