import torch, ttnn, sys

sys.path.insert(0, "tests/ttnn/unit_tests/operations/matmul_reduce_scatter")
from tests.scripts.common import get_updated_device_params
from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter
import test_matmul_reduce_scatter_precision_baseline as P

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D, ttnn.FabricReliabilityMode.STRICT_INIT)
params = get_updated_device_params({"fabric_config": ttnn.FabricConfig.FABRIC_2D})
params.pop("fabric_config")
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 4), **params)
try:
    a_shape, w_shape = (1, 1, 640, 2048), (2048, 7168)
    for K in (2048, 512):
        a_shape, w_shape = (1, 1, 640, K), (K, 7168)
        a = P._stacked_randn(mesh, a_shape, 0)
        w = P._as_device_holds(P._stacked_randn(mesh, w_shape, 1, scale=K**-0.5), ttnn.bfloat8_b)
        exp = P._reference(a, w, 1, -1)
        # single-device partial reference (bf16-rounded partials), to separate matmul vs transport
        for fp32 in (False, True):
            out = matmul_reduce_scatter(
                P._to_mesh(a, mesh, ttnn.bfloat16),
                P._to_mesh(w, mesh, ttnn.bfloat8_b),
                cluster_axis=1,
                scatter_dim=-1,
                num_links=2,
                compute_kernel_config=ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32
                ),
            )
            t = ttnn.get_device_tensors(out)[3]
            act = ttnn.to_torch(t).double().flatten()
            e = exp[0, 3].double().flatten()
            slope = float((act * e).sum() / (e * e).sum())
            big = e.abs() > e.pow(2).mean().sqrt()
            r = act[big] / e[big]
            pos = e > 0
            print(
                f"PROBE K={K} fp32={fp32} slope={slope:.5f} ratio_big_median={float(r.median()):.5f} p5={float(r.quantile(0.05)):.4f} p95={float(r.quantile(0.95)):.4f} "
                f"mean_err_pos={float((act-e)[pos].mean()):.5f} mean_err_neg={float((act-e)[~pos].mean()):.5f} rel_rms={float(((act-e)**2).mean().sqrt()/(e**2).mean().sqrt()):.5f}"
            )
finally:
    ttnn.close_mesh_device(mesh)
