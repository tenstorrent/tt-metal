import torch, ttnn
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import pytorch_groupnorm_sc_N_1_HW_C
from ttnn.operations.groupnorm_sc_N_1_HW_C_b import groupnorm_sc_N_1_HW_C_b

device = ttnn.open_device(device_id=0)
TD = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}
try:
    shapes = [
        ((1, 1, 64, 17), 1),
        ((1, 1, 64, 50), 1),
        ((1, 1, 128, 100), 1),
        ((2, 1, 64, 47), 1),
        ((1, 1, 64, 48), 2),
        ((1, 1, 64, 80), 2),
        ((1, 1, 128, 48), 3),
        ((1, 1, 64, 80), 4),
        ((1, 1, 128, 144), 4),
        ((1, 1, 64, 200), 8),
        ((2, 1, 64, 48), 2),
    ]
    for dt in (ttnn.bfloat16, ttnn.float32):
        worst = (1.0, 0.0)
        for shape, G in shapes:
            for layout in (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT):
                torch.manual_seed(0)
                x = torch.randn(shape).to(TD[dt])
                C = shape[-1]
                g = torch.randn(1, 1, 1, C).to(TD[dt])
                b = torch.randn(1, 1, 1, C).to(TD[dt])
                exp = pytorch_groupnorm_sc_N_1_HW_C(x, G, gamma=g, beta=b).float()
                tx = ttnn.from_torch(x, dtype=dt, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                tg = ttnn.from_torch(
                    g, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                tb = ttnn.from_torch(
                    b, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                y = ttnn.to_torch(groupnorm_sc_N_1_HW_C_b(tx, G, gamma=tg, beta=tb)).float()
                pcc = torch.corrcoef(torch.stack([y.flatten(), exp.flatten()]))[0, 1].item()
                rms = ((y - exp).pow(2).mean().sqrt() / exp.pow(2).mean().sqrt()).item()
                worst = (min(worst[0], pcc), max(worst[1], rms))
        print(f"ACC {dt}: min PCC {worst[0]:.6f}  max relRMS {worst[1]:.2e}")
finally:
    ttnn.close_device(device)
