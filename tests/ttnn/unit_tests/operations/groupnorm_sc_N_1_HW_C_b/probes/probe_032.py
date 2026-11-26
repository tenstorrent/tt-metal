import torch, ttnn
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import pytorch_groupnorm_sc_N_1_HW_C
from ttnn.operations.groupnorm_sc_N_1_HW_C_b import groupnorm_sc_N_1_HW_C_b

device = ttnn.open_device(device_id=0)
try:
    dt = ttnn.float32
    for C in (32, 47, 50, 64, 96, 100, 128):
        r = []
        for seed in range(6):
            torch.manual_seed(seed)
            x = torch.randn(1, 1, 64, C)
            x[0, 0, 0, 0] = 2.84
            exp = pytorch_groupnorm_sc_N_1_HW_C(x, 1).float()
            tx = ttnn.from_torch(
                x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            y = ttnn.to_torch(groupnorm_sc_N_1_HW_C_b(tx, 1)).float()
            e = y - exp
            r.append((e.pow(2).mean().sqrt() / exp.pow(2).mean().sqrt()).item())
        print(
            f"ACC C={C:4d} k=+2.84 relRMS per seed: " + " ".join(f"{v:.1e}" for v in r) + f"  mean {sum(r)/len(r):.2e}"
        )
finally:
    ttnn.close_device(device)
