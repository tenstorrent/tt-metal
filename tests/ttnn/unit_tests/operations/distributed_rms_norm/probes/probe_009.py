import torch, ttnn, gc
from eval.golden_tests.distributed_rms_norm.conftest import production_router_config
from eval.golden_tests.distributed_rms_norm.helpers import run_distributed_rms_norm
import ttnn.operations.distributed_rms_norm.distributed_rms_norm_program_descriptor as pd
from ttnn.operations.ccl import Topology

orig_alloc = ttnn.allocate_tensor_on_device


def logged_alloc(*a):
    t = orig_alloc(*a)
    print("ALLOC", a[0], a[4].memory_layout, t.buffer_address())
    return t


pd.ttnn.allocate_tensor_on_device = logged_alloc
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
H = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
W = ttnn.TensorMemoryLayout.WIDTH_SHARDED


def go(shape, ml, dt, gd):
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


try:
    go((1, 1, 640, 1792), H, ttnn.bfloat16, ttnn.bfloat16)
    print("PASS a")
    go((1, 1, 640, 1792), H, ttnn.bfloat16, ttnn.bfloat16)
    print("PASS b")
    go((1, 1, 32, 1792), W, ttnn.bfloat16, ttnn.bfloat16)
    print("PASS c")
    go((1, 1, 32, 1792), W, ttnn.bfloat16, ttnn.bfloat16)
    print("PASS d")
finally:
    ttnn.close_mesh_device(mesh)
