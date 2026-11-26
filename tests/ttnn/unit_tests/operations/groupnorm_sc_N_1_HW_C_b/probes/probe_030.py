import torch, ttnn
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import pytorch_groupnorm_sc_N_1_HW_C
from ttnn.operations.groupnorm_sc_N_1_HW_C_b import groupnorm_sc_N_1_HW_C_b

device = ttnn.open_device(device_id=0)
try:
    dt = ttnn.float32
    for C in (47, 64):
        for layout in (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT):
            out = []
            for seed in range(5):
                torch.manual_seed(seed)
                shape = (2, 1, 64, C)
                x = torch.randn(shape)
                g = torch.randn(1, 1, 1, C)
                b = torch.randn(1, 1, 1, C)
                exp = pytorch_groupnorm_sc_N_1_HW_C(x, 1, gamma=g, beta=b).float()
                tx = ttnn.from_torch(x, dtype=dt, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                tg = ttnn.from_torch(
                    g, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                tb = ttnn.from_torch(
                    b, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                y = ttnn.to_torch(groupnorm_sc_N_1_HW_C_b(tx, 1, gamma=tg, beta=tb)).float()
                per = []
                for n in range(2):
                    e = y[n] - exp[n]
                    per.append(
                        f"{(e.pow(2).mean().sqrt()/exp[n].pow(2).mean().sqrt()).item():.1e}(k={x[n,0,0,0].item():+.2f})"
                    )
                out.append(" ".join(per))
            print(f"ACC C={C} {layout}: " + " | ".join(out))
finally:
    ttnn.close_device(device)
