import sys, torch, ttnn
from eval.golden_tests.distributed_rms_norm.helpers import create_ttnn_input_tensor, random_stacked
from eval.golden_tests.distributed_rms_norm.topology import FABRIC_CONFIG
from ttnn.operations.distributed_rms_norm import distributed_rms_norm
from tests.scripts.common import get_updated_device_params

RUN_OP = 0


def open_mesh(shape):
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
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*shape), **params)


def close(mesh):
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


for i, (shape, axis) in enumerate([((2, 2), 1), ((1, 4), 1), ((4, 1), 0), ((1, 4), 1)]):
    m = open_mesh(shape)
    if RUN_OP:
        x = random_stacked(m, (1, 1, 64, 128), torch.bfloat16, seed=1)
        tx = create_ttnn_input_tensor(x, m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        distributed_rms_norm(tx, cluster_axis=axis)
        ttnn.synchronize_device(m)
    print("PROBE step", i, shape, "ok", flush=True)
    close(m)
