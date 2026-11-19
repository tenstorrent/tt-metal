import os, sys, torch, ttnn

sys.path.insert(0, os.getcwd())
from eval.golden_tests.all_gather_rms_norm.helpers import (
    create_ttnn_input_tensor,
    create_ttnn_gamma,
    random_stacked,
    random_gamma,
)
from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm

shape = tuple(int(v) for v in os.environ.get("AGZ_SHAPE", "1,1,640,384").split(","))
ln = os.environ.get("AGZ_LN") == "1"
cfg = ttnn._ttnn.fabric.FabricRouterConfig()
cfg.max_packet_payload_size_bytes = 14 * 1024 + 64
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D_TORUS_XY, router_config=cfg)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
try:
    x = random_stacked(mesh, shape, torch.bfloat16, seed=1)
    W = 4 * shape[-1]
    g = create_ttnn_gamma(random_gamma(W, torch.bfloat16), mesh, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    b = (
        create_ttnn_gamma(
            random_gamma(W, torch.bfloat16, seed=3), mesh, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        if ln
        else None
    )
    tx = create_ttnn_input_tensor(x, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    kw = dict(norm_type="layer_norm", beta=b) if ln else {}
    for i in range(4):
        out = all_gather_rms_norm(tx, cluster_axis=1, gamma=g, topology=ttnn.Topology.Ring, num_links=2, **kw)
        ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    print("AGZ_DONE")
finally:
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
