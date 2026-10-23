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
m = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(2, 2), **params)


def run(shape):
    x = random_stacked(m, shape, torch.bfloat16, seed=5)
    exp = pytorch_distributed_rms_norm(x, None, 1, 1e-6)
    tx = create_ttnn_input_tensor(x, m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    out = distributed_rms_norm(tx, cluster_axis=1, num_links=1)
    a = ttnn.to_torch(ttnn.get_device_tensors(out)[1]).float()
    e = exp[0, 1].float()
    return float(((a - e) ** 2).mean().sqrt() / (e**2).mean().sqrt())


shapes = [
    (1, 1, 32, 1792),
    (1, 1, 640, 1792),
    (1, 1, 2048, 1280),
    (1, 1, 4096, 1024),
    (2, 4, 128, 512),
    (1, 1, 128, 4096),
    (1, 1, 1024, 512),
    (1, 1, 3072, 512),
]
for s in shapes:
    print(
        "PROBE",
        s,
        "rel_rms %.4f" % run(s),
        "buckets",
        {k[1]: sorted(e["scratch"]) for k, e in pd._PERSIST.items()},
        flush=True,
    )
for e in pd._PERSIST.values():
    for t in e["scratch"].values():
        ttnn.deallocate(t)
print(
    "PROBE after external dealloc",
    (1, 1, 640, 1792),
    "rel_rms %.4f" % run((1, 1, 640, 1792)),
    {k[1]: sorted(e["scratch"]) for k, e in pd._PERSIST.items()},
    flush=True,
)
ttnn.close_mesh_device(m)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
