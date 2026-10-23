import torch, ttnn
from eval.golden_tests.distributed_rms_norm.helpers import (
    create_ttnn_input_tensor,
    pytorch_distributed_rms_norm,
    random_stacked,
)
from eval.golden_tests.distributed_rms_norm.topology import FABRIC_CONFIG
from ttnn.operations.distributed_rms_norm import distributed_rms_norm
from ttnn.operations.distributed_rms_norm import distributed_rms_norm_program_descriptor as pd
from tests.scripts.common import get_updated_device_params

cfg = ttnn._ttnn.fabric.FabricRouterConfig()
cfg.max_packet_payload_size_bytes = 14 * 1024 + 64
ttnn.set_fabric_config(
    FABRIC_CONFIG,
    ttnn.FabricReliabilityMode.STRICT_INIT,
    None,
    ttnn.FabricTensixConfig.DISABLED,
    ttnn.FabricUDMMode.DISABLED,
    ttnn.FabricManagerMode.DEFAULT,
    cfg,
)
params = get_updated_device_params({"fabric_config": FABRIC_CONFIG})
params.pop("fabric_config")
mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 2), **params)
shape = (1, 1, 64, 128)
x = random_stacked(mesh, shape, torch.bfloat16, seed=1)
exp = pytorch_distributed_rms_norm(x, None, 1, 1e-6)
tx = create_ttnn_input_tensor(x, mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
ttnn.graph.up_front_begin_collect(clear=False, real_alloc=False)
try:
    distributed_rms_norm(tx, cluster_axis=1)
finally:
    ttnn.graph.up_front_end_collect()
for k, e in pd._PERSIST.items():
    print(
        "PROBE cached",
        k,
        [int(ttnn.get_global_semaphore_address(s)) for s in e["sems"]],
        {b: int(t.buffer_address()) for b, t in e["scratch"].items()},
    )
out = distributed_rms_norm(tx, cluster_axis=1)
d = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
print(
    "PROBE after-collect real call max|out|",
    float(d.abs().max()),
    "max err",
    float((d - exp[0, 0].float()).abs().max()),
)
pd._PERSIST.clear()
out = distributed_rms_norm(tx, cluster_axis=1)
d = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
print(
    "PROBE fresh-cache real call max|out|", float(d.abs().max()), "max err", float((d - exp[0, 0].float()).abs().max())
)
for k, e in pd._PERSIST.items():
    print(
        "PROBE cached",
        k,
        [int(ttnn.get_global_semaphore_address(s)) for s in e["sems"]],
        {b: int(t.buffer_address()) for b, t in e["scratch"].items()},
    )
ttnn.close_mesh_device(mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
