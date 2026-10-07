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
from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter as mod_fn
import sys

M = sys.modules["ttnn.operations.matmul_reduce_scatter.matmul_reduce_scatter"]
fc = ttnn.FabricConfig.FABRIC_2D
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
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 4), **p)
try:
    for axis, a_shape, w_shape, sd, L in [
        (1, (1, 1, 640, 2048), (2048, 7168), -1, 2),
        (0, (1, 1, 256, 512), (512, 256), -2, 1),
        (1, (1, 1, 2048, 2048), (2048, 4096), -2, 2),
    ]:
        a = random_stacked(mesh, a_shape, torch.bfloat16, seed=3)
        w = quantize_like_device(
            random_stacked(mesh, w_shape, torch.bfloat16, seed=4, scale=w_shape[0] ** -0.5), ttnn.bfloat8_b
        )
        exp = pytorch_matmul_reduce_scatter(a, w, axis, sd)
        ta = create_ttnn_input_tensor(a, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        tw = create_ttnn_input_tensor(w, mesh, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        for it in range(2):
            o = matmul_reduce_scatter(ta, tw, cluster_axis=axis, scatter_dim=sd, num_links=L)
            dev = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(o)]
            rows, cols = tuple(mesh.shape)
            worst = min(_pcc_rel_rms(d, exp[i // cols, i % cols].float())[0] for i, d in enumerate(dev))
            print(f"RES axis={axis} {a_shape} sd={sd} L={L} it={it} worst_pcc={worst:.6f}")
        for k, (_, plan) in M._PLAN_CACHE.items():
            if k[1] == axis and k[3] == L:
                pl = plan["pl"]
                print("PL", axis, L, pl.mode)
                for c in sorted(pl.fwd_ports, key=str):
                    f = lambda lst: [(q.x, q.y) for q in lst]
                    print("PL", c, "fwd", f(pl.fwd_ports[c]), "bwd", f(pl.bwd_ports[c]), "fin", f(pl.finals[c]))
                break
finally:
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
