import torch, ttnn
from eval.golden_tests.distributed_rms_norm.helpers import (
    create_ttnn_input_tensor,
    create_ttnn_gamma,
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


def run(shape, cap=None, gamma=True, reps=3):
    pd.BLOCK_ROWS_CAP = cap
    x = random_stacked(m, shape, torch.bfloat16, seed=3)
    g = random_stacked(m, (1, 1, 1, shape[-1]), torch.bfloat16, seed=4) if gamma else None
    exp = pytorch_distributed_rms_norm(x, g, 1, 1e-6)
    tx = create_ttnn_input_tensor(x, m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tg = create_ttnn_gamma(g, m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) if gamma else None
    plan = pd.make_plan(m, tx, tg, 1, ttnn.Topology.Linear, 1)
    for r in range(reps):
        out = distributed_rms_norm(tx, cluster_axis=1, gamma=tg, num_links=1)
        errs = []
        for i, d in enumerate(ttnn.get_device_tensors(out)):
            a = ttnn.to_torch(d).float()
            e = exp[i // 2, i % 2].float()
            errs.append(float(((a - e) ** 2).mean().sqrt() / (e**2).mean().sqrt()))
        print(
            f"PROBE shape={shape} cap={cap} gamma={gamma} regime={plan.regime} R={plan.R} waves={plan.num_waves} WS={plan.WS} C={plan.C} nch={plan.nch} rep={r} rel_rms={['%.3g'%v for v in errs]}",
            flush=True,
        )


run((2, 1, 2048, 4096))
run((1, 1, 4096, 4096))
run((1, 1, 2048, 4096), cap=None)
run((1, 1, 4096, 2048), cap=1)
run((1, 1, 4096, 4096), gamma=False)
ttnn.close_mesh_device(m)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
