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


def run(shape, cap, reps=2):
    pd.BLOCK_ROWS_CAP = cap
    x = random_stacked(m, shape, torch.bfloat16, seed=3)
    exp = pytorch_distributed_rms_norm(x, None, 1, 1e-6)
    tx = create_ttnn_input_tensor(x, m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    plan = pd.make_plan(m, tx, None, 1, ttnn.Topology.Linear, 1)
    row2 = {}
    for g in plan.groups:
        for k in range(g.rows):
            row2[g.start_row + k] = (k // plan.R, g.cores[0])
    for r in range(reps):
        out = distributed_rms_norm(tx, cluster_axis=1, num_links=1)
        for i, d in enumerate(ttnn.get_device_tensors(out)):
            a = ttnn.to_torch(d).float().reshape(-1, shape[-1])
            e = exp[i // 2, i % 2].float().reshape(-1, shape[-1])
            Ht = a.shape[0] // 32
            bad = [t for t in range(Ht) if (a[t * 32 : (t + 1) * 32] - e[t * 32 : (t + 1) * 32]).abs().max() > 0.1]
            if bad:
                waves = sorted(set(row2[t][0] for t in bad))
                cores = sorted(set((row2[t][1].x, row2[t][1].y) for t in bad))
                rr = (a[bad[0] * 32] / e[bad[0] * 32])[e[bad[0] * 32].abs() > 0.3]
                print(
                    f"PROBE shape={shape} cap={cap} R={plan.R} nw={plan.num_waves} groups={len(plan.groups)} rep={r} dev={i} bad_rows={len(bad)}/{Ht} waves={waves} ncores={len(cores)} first={cores[:4]} ratio[{rr.min():.3f},{rr.max():.3f}]",
                    flush=True,
                )
        print(f"PROBE shape={shape} cap={cap} rep={r} done", flush=True)


run((1, 1, 128, 512), 1)
run((1, 1, 256, 512), 1)
run((1, 1, 256, 512), 2)
run((1, 1, 128, 1024), 1)
run((1, 1, 512, 512), 2)
ttnn.close_mesh_device(m)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
