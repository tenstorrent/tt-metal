import torch, ttnn
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import pytorch_groupnorm_sc_N_1_HW_C
from ttnn.operations.groupnorm_sc_N_1_HW_C_b import groupnorm_sc_N_1_HW_C_b

device = ttnn.open_device(device_id=0)
try:
    dt = ttnn.float32
    torch.manual_seed(0)
    base = torch.randn(2, 1, 64, 64)
    g0 = torch.randn(1, 1, 1, 64)
    b0 = torch.randn(1, 1, 1, 64)
    for kval in (0.0, 2.84, -2.84):
        for C in (47, 64):
            x = base[..., :C].clone()
            x[1, 0, 0, 0] = kval
            g = g0[..., :C]
            b = b0[..., :C]
            exp = pytorch_groupnorm_sc_N_1_HW_C(x, 1, gamma=g, beta=b).float()
            tx = ttnn.from_torch(
                x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            tg = ttnn.from_torch(
                g.contiguous(),
                dtype=dt,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            tb = ttnn.from_torch(
                b.contiguous(),
                dtype=dt,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            y = ttnn.to_torch(groupnorm_sc_N_1_HW_C_b(tx, 1, gamma=tg, beta=tb)).float()
            e = y[1] - exp[1]
            print(
                f"ACC k={kval:+.2f} C={C}: image1 relRMS {(e.pow(2).mean().sqrt()/exp[1].pow(2).mean().sqrt()).item():.2e}"
            )
finally:
    ttnn.close_device(device)
