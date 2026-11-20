import torch, ttnn, sys

sys.path.insert(0, "tests/ttnn/unit_tests/operations")
from all_gather_rms_norm.test_all_gather_rms_norm import _DEVICE_PARAMS, make_input
from eval.golden_tests.all_gather_rms_norm.helpers import create_ttnn_input_tensor
from ttnn.operations.all_gather_rms_norm import all_gather_rms_norm

p = _DEVICE_PARAMS[0]
ttnn.set_fabric_config(
    p["fabric_config"],
    ttnn.FabricReliabilityMode.STRICT_SYSTEM_HEALTH_SETUP_MODE,
    None,
    ttnn.FabricTensixConfig.DISABLED,
    ttnn.FabricUDMMode.DISABLED,
    ttnn.FabricManagerMode.DEFAULT,
    p["fabric_router_config"],
)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
x = make_input((1, 4), (1, 1, 2048, 384), ttnn.bfloat16, seed=11)
tx = create_ttnn_input_tensor(x, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
ckc = ttnn.ComputeConfigDescriptor()
ckc.math_fidelity = ttnn.MathFidelity.HiFi4
ckc.fp32_dest_acc_en = False
out = all_gather_rms_norm(
    tx,
    cluster_axis=1,
    epsilon=1e-6,
    topology=ttnn.Topology.Ring,
    num_links=2,
    norm_type="layer_norm",
    compute_kernel_config=ckc,
)
a = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().reshape(-1, 1536)
bad = (~torch.isfinite(a)).any(-1).reshape(-1, 32).any(-1)
print("ROWS bad tile-rows:", bad.nonzero().flatten().tolist()[:40], "of", bad.numel())
badc = (~torch.isfinite(a)).any(0).reshape(-1, 32).any(-1)
print("ROWS bad tile-cols:", badc.nonzero().flatten().tolist())
ttnn.close_mesh_device(mesh)
