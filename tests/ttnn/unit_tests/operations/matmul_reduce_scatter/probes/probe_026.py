import torch, ttnn
from tests.scripts.common import get_updated_device_params
from eval.golden_tests.matmul_reduce_scatter.conftest import production_router_config
from eval.golden_tests.matmul_reduce_scatter.helpers import (
    random_stacked,
    quantize_like_device,
    pytorch_matmul_reduce_scatter,
    create_ttnn_input_tensor,
    _pcc_rel_rms,
)
from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter

fc = ttnn.FabricConfig.FABRIC_2D_TORUS_X
ttnn.set_fabric_config(
    fc,
    ttnn.FabricReliabilityMode.STRICT_INIT,
    None,
    ttnn.FabricTensixConfig.DISABLED,
    ttnn.FabricUDMMode.DISABLED,
    ttnn.FabricManagerMode.DEFAULT,
    production_router_config(),
)
p = get_updated_device_params({"fabric_config": fc})
p.pop("fabric_config")
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 8), **p)
try:
    for a_shape, w_shape, sd in [
        ((1, 1, 640, 2048), (2048, 7168), -1),
        ((1, 1, 256, 256), (256, 256), -2),
        ((1, 1, 2048, 2048), (2048, 4096), -2),
    ]:
        a = random_stacked(mesh, a_shape, torch.bfloat16, seed=3)
        w = quantize_like_device(
            random_stacked(mesh, w_shape, torch.bfloat16, seed=4, scale=w_shape[0] ** -0.5), ttnn.bfloat8_b
        )
        exp = pytorch_matmul_reduce_scatter(a, w, 1, sd)
        ta = create_ttnn_input_tensor(a, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        tw = create_ttnn_input_tensor(w, mesh, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        outs = []
        for it, topo in enumerate(
            [
                ttnn.Topology.Ring,
                ttnn.Topology.Linear,
                ttnn.Topology.Ring,
                ttnn.Topology.Ring,
                ttnn.Topology.Linear,
                ttnn.Topology.Ring,
            ]
        ):
            o = matmul_reduce_scatter(ta, tw, cluster_axis=1, scatter_dim=sd, topology=topo, num_links=2)
            dev = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(o)]
            worst = min(_pcc_rel_rms(d, exp[0, c].float())[0] for c, d in enumerate(dev))
            outs.append((topo, dev))
            print(f"RES {a_shape} sd={sd} it={it} {topo} worst_pcc={worst:.6f}")
        rings = [d for t, d in outs if t == ttnn.Topology.Ring]
        same = all(torch.equal(x, y) for r in rings[1:] for x, y in zip(rings[0], r))
        print(f"RES {a_shape} ring bit-exact across {len(rings)} calls: {same}")
finally:
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
