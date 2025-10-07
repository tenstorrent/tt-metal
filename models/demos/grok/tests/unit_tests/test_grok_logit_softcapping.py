import torch

import ttnn


def test_grok_logit_softcapping(mesh_device):
    torch_input = torch.randn([1, 1, 32, 8]).to(torch.float32)
    g = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ref_output = 30.0 * torch.tanh(torch_input / 30.0)

    g = ttnn.div(g, 30.0)
    g = ttnn.tanh(g)
    g = ttnn.mul(g, 30.0)
    torch_g = ttnn.to_torch(g, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1, :, :, :]

    breakpoint()
