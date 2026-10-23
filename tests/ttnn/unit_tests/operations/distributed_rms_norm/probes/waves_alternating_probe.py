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


def alt(shape, cap, n=4):
    pd.BLOCK_ROWS_CAP = cap
    ins = []
    for s in (11, 12):
        x = random_stacked(m, shape, torch.bfloat16, seed=s)
        ins.append(
            (
                create_ttnn_input_tensor(x, m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
                pytorch_distributed_rms_norm(x, None, 1, 1e-6),
            )
        )
    plan = pd.make_plan(m, ins[0][0], None, 1, ttnn.Topology.Linear, 1)
    for k in range(n):
        tx, exp = ins[k % 2]
        out = distributed_rms_norm(tx, cluster_axis=1, num_links=1)
        res = []
        for i, d in enumerate(ttnn.get_device_tensors(out)):
            a = ttnn.to_torch(d).float().reshape(-1, shape[-1])
            e = exp[i // 2, i % 2].float().reshape(-1, shape[-1])
            Ht = a.shape[0] // 32
            bad = [t for t in range(Ht) if (a[t * 32 : (t + 1) * 32] - e[t * 32 : (t + 1) * 32]).abs().max() > 0.1]
            # segment position of each bad row: wave-0 row of group gi sits at offset gi*128
            pos = [gi for gi, g in enumerate(plan.groups) if g.start_row in bad]
            res.append((len(bad), pos[:3], pos[-3:] if pos else []))
        print(
            f"PROBE shape={shape} cap={cap} R={plan.R} nw={plan.num_waves} regime={plan.regime} call={k} input={k%2} bad_rows_per_dev={res}",
            flush=True,
        )


alt((1, 1, 4096, 1024), 1)
alt((2, 1, 2048, 4096), None)
alt((1, 1, 4096, 2048), 1)
alt((1, 1, 640, 1792), 1)
alt((1, 1, 4096, 1024), None)
ttnn.close_mesh_device(m)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
