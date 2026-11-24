import torch, ttnn
from eval.golden_tests.all_gather_rms_norm.helpers import (
    create_ttnn_input_tensor,
    create_ttnn_gamma,
    pytorch_all_gather_rms_norm,
)
from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D_TORUS_XY)
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
for shape, norm in [((1, 1, 640, 384), "rms_norm"), ((1, 1, 640, 32), "layer_norm")]:
    torch.manual_seed(0)
    x = torch.randn((1, 4, *shape)).to(torch.bfloat16)
    g = torch.randn(4 * shape[-1]).to(torch.bfloat16)
    b = torch.randn(4 * shape[-1]).to(torch.bfloat16) if norm == "layer_norm" else None
    kw = dict(gamma=create_ttnn_gamma(g, mesh, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT))
    if b is not None:
        kw["beta"] = create_ttnn_gamma(b, mesh, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    exp = pytorch_all_gather_rms_norm(x, g, 1, 1e-6, norm_type=norm, beta=b)
    o = all_gather_rms_norm(
        create_ttnn_input_tensor(x, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        cluster_axis=1,
        norm_type=norm,
        topology=ttnn.Topology.Ring,
        num_links=2,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        **kw,
    )
    worst = (1, 0)
    for i, t in enumerate(ttnn.get_device_tensors(o)):
        a = ttnn.to_torch(t).float().flatten()
        e = exp[0, i].float().flatten()
        pcc = torch.corrcoef(torch.stack([a, e]))[0, 1].item()
        rr = ((a - e).pow(2).mean().sqrt() / e.pow(2).mean().sqrt()).item()
        worst = (min(worst[0], pcc), max(worst[1], rr))
    print(f"ACC {shape} {norm} min PCC={worst[0]:.6f} max relRMS={worst[1]:.5f}")
ttnn.close_mesh_device(mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
