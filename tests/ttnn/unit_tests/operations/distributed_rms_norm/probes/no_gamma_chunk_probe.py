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


def run(shape, cap=None, gamma=False):
    pd.BLOCK_ROWS_CAP = cap
    x = random_stacked(m, shape, torch.bfloat16, seed=3)
    g = random_stacked(m, (1, 1, 1, shape[-1]), torch.bfloat16, seed=4) if gamma else None
    exp = pytorch_distributed_rms_norm(x, g, 1, 1e-6)
    tx = create_ttnn_input_tensor(x, m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tg = create_ttnn_gamma(g, m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) if gamma else None
    plan = pd.make_plan(m, tx, tg, 1, ttnn.Topology.Linear, 1)
    out = distributed_rms_norm(tx, cluster_axis=1, gamma=tg, num_links=1)
    a = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().reshape(-1, shape[-1])
    e = exp[0, 0].float().reshape(-1, shape[-1])
    Ht, Wt = a.shape[0] // 32, a.shape[1] // 32
    bad = torch.zeros(Ht, Wt, dtype=torch.bool)
    for i in range(Ht):
        for j in range(Wt):
            ta = a[i * 32 : (i + 1) * 32, j * 32 : (j + 1) * 32]
            te = e[i * 32 : (i + 1) * 32, j * 32 : (j + 1) * 32]
            bad[i, j] = (ta - te).abs().max() > 0.1
    rows_bad = bad.any(1).nonzero().flatten().tolist()
    cols_bad = bad.any(0).nonzero().flatten().tolist()
    ratio = (a / e)[e.abs() > 0.5]
    print(
        f"PROBE shape={shape} gamma={gamma} R={plan.R} waves={plan.num_waves} WS={plan.WS} C={plan.C} nch={plan.nch} bad_tiles={int(bad.sum())}/{Ht*Wt} bad_rows={rows_bad[:12]}.. n={len(rows_bad)} bad_cols={cols_bad[:12]}.. n={len(cols_bad)} ratio_med={ratio.median():.4f}",
        flush=True,
    )
    if bad.any():
        i = rows_bad[0]
        j = cols_bad[0]
        print("PROBE sample a", a[i * 32, j * 32 : j * 32 + 4].tolist(), "e", e[i * 32, j * 32 : j * 32 + 4].tolist())


run((1, 1, 4096, 4096))
run((1, 1, 4096, 4096), gamma=True)
run((1, 1, 2048, 4096))
run((1, 1, 4096, 2560))
ttnn.close_mesh_device(m)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
