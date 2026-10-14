import torch, ttnn
from eval.golden_tests.distributed_rms_norm.conftest import production_router_config
from eval.golden_tests.distributed_rms_norm.helpers import run_distributed_rms_norm
import ttnn.operations.distributed_rms_norm.distributed_rms_norm_program_descriptor as pd
from ttnn.operations.ccl import Topology

orig = pd._choose_knobs


def logged(plan, x_dtype, g_dtype, out_native, budget):
    r = orig(plan, x_dtype, g_dtype, out_native, budget)
    print(
        "KNOBS",
        "S_r",
        plan["S_r"],
        "S_w",
        plan["S_w"],
        "rows",
        plan["block_rows"],
        "cw",
        plan["core_w_tiles"],
        "lc",
        plan["local_combine"],
        max(plan["lc_per"]),
        "native",
        out_native,
        "budget",
        budget,
        "-> stat_rows,gc,prescale",
        r,
        "ports",
        [(c.x, c.y) for c in plan["port_cores"]],
    )
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
H = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
W = ttnn.TensorMemoryLayout.WIDTH_SHARDED
try:
    for shape, ml, dt, gd in [
        ((1, 1, 32, 8192), H, ttnn.bfloat16, ttnn.float32),
        ((1, 1, 128, 4096), H, ttnn.float32, ttnn.float32),
        ((1, 1, 2048, 2048), W, ttnn.bfloat16, ttnn.bfloat16),
        ((1, 1, 4096, 1024), W, ttnn.float32, ttnn.float32),
        ((1, 1, 32, 8192), W, ttnn.bfloat16, "none"),
    ]:
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
        import gc

        gc.collect()
finally:
    ttnn.close_mesh_device(mesh)
