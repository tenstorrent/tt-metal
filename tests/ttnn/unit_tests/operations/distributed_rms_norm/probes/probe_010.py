import torch, ttnn, gc
from eval.golden_tests.distributed_rms_norm.conftest import production_router_config
from eval.golden_tests.distributed_rms_norm.helpers import run_distributed_rms_norm
import ttnn.operations.distributed_rms_norm.distributed_rms_norm_program_descriptor as pd
from ttnn.operations.ccl import Topology

orig_alloc = ttnn.allocate_tensor_on_device
orig_run = pd.run_distributed_rms_norm


def run2(x, *a, **k):
    print("XADDR", x.buffer_address(), x.memory_config())
    return orig_run(x, *a, **k)


import ttnn.operations.distributed_rms_norm.distributed_rms_norm as opm

opm.run_distributed_rms_norm = run2


def logged_alloc(*a):
    t = orig_alloc(*a)
    print("ALLOC", a[0], a[4].memory_layout, t.buffer_address())
    return t


pd.ttnn.allocate_tensor_on_device = logged_alloc
orig = pd._choose_knobs


def logged(plan, x_dtype, g_dtype, out_native, budget):
    r = orig(plan, x_dtype, g_dtype, out_native, budget)
    print("KNOBS budget", budget, r, pd._worker_footprint(plan, x_dtype, g_dtype, r[2], r[0], r[1], out_native))
    return r


pd._choose_knobs = logged
ttnn.set_fabric_config(
    ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    ttnn.FabricReliabilityMode.STRICT_INIT,
    None,
    ttnn.FabricTensixConfig.DISABLED,
    ttnn.FabricUDMMode.DISABLED,
    ttnn.FabricManagerMode.DEFAULT,
    production_router_config(),
)
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 2))
W = ttnn.TensorMemoryLayout.WIDTH_SHARDED
try:
    for shape, ml, dt, gd in [((1, 1, 128, 4096), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.bfloat16, ttnn.float32)]:
        run_distributed_rms_norm(
            (shape,),
            mesh_device=mesh,
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            fp32_dest_acc_en=True,
            gamma_mode="no_gamma" if gd == "none" else "gamma",
            gamma_dtype=gd,
            memory_layout=ml,
            cluster_axis=1,
            topology=Topology.Linear,
            num_links=None,
        )
        print("PASS", shape, ml)
finally:
    ttnn.close_mesh_device(mesh)
