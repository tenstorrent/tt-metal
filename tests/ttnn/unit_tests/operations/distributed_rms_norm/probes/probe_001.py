import torch, ttnn
from eval.golden_tests.distributed_rms_norm.conftest import production_router_config
from eval.golden_tests.distributed_rms_norm.helpers import (
    check_output,
    create_ttnn_input_tensor,
    pytorch_distributed_rms_norm,
    random_stacked,
)
from ttnn.operations.distributed_rms_norm import distributed_rms_norm
from ttnn.operations.ccl import Topology

ttnn.set_fabric_config(
    ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
    ttnn.FabricReliabilityMode.STRICT_INIT,
    None,
    ttnn.FabricTensixConfig.DISABLED,
    ttnn.FabricUDMMode.DISABLED,
    ttnn.FabricManagerMode.DEFAULT,
    production_router_config(),
)
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
try:
    for dtype, tdt, with_g, shape in [
        (ttnn.float32, torch.float32, False, (1, 1, 64, 1024)),
        (ttnn.bfloat16, torch.bfloat16, False, (1, 1, 32, 1792)),
        (ttnn.float32, torch.float32, True, (2, 1, 128, 512)),
        (ttnn.bfloat16, torch.bfloat16, True, (1, 1, 2048, 1280)),
    ]:
        x = random_stacked(mesh, shape, tdt, seed=3)
        g = random_stacked(mesh, (1, 1, 1, shape[-1]), tdt, seed=4) if with_g else None
        exp = pytorch_distributed_rms_norm(x, g, 1, 1e-6)
        tx = create_ttnn_input_tensor(x, mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        tg = create_ttnn_input_tensor(g, mesh, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT) if with_g else None
        outs = []
        for rep in range(3):
            o = distributed_rms_norm(tx, cluster_axis=1, gamma=tg, topology=Topology.Ring, num_links=2)
            check_output(o, exp, shape=shape, dtype=dtype, fp32_dest_acc_en=True, expected_layout=ttnn.TILE_LAYOUT)
            outs.append([ttnn.to_torch(t) for t in ttnn.get_device_tensors(o)])
        ident = all(torch.equal(a, b) for r in outs[1:] for a, b in zip(outs[0], r))
        print("RING ok", dtype, with_g, shape, "bit_identical_repeats", ident)
finally:
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
