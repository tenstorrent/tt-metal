import torch, ttnn
from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import pytorch_groupnorm_sc_N_1_HW_C
from ttnn.operations.groupnorm_sc_N_1_HW_C_b import groupnorm_sc_N_1_HW_C_b

device = ttnn.open_device(device_id=0)
TD = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}
try:
    shapes = [
        ((1, 1, 64, 17), 1),
        ((1, 1, 64, 32), 1),
        ((1, 1, 64, 50), 1),
        ((1, 1, 64, 64), 1),
        ((1, 1, 128, 100), 1),
        ((1, 1, 128, 128), 1),
        ((1, 1, 64, 200), 8),
        ((1, 1, 64, 256), 8),
        ((1, 1, 64, 48), 2),
        ((1, 1, 64, 64), 2),
    ]
    dt = ttnn.float32
    for shape, G in shapes:
        for layout in (ttnn.TILE_LAYOUT,):
            for seed in (0, 1):
                torch.manual_seed(seed)
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
                rms = ((y - exp).pow(2).mean().sqrt() / exp.pow(2).mean().sqrt()).item()
                # stats-only error: compare normalized (no affine) via mean/rstd recovery is complex; report max abs too
                print(f"ACC {shape} G{G} seed{seed}: relRMS {rms:.2e} maxabs {(y-exp).abs().max().item():.2e}")
finally:
    ttnn.close_device(device)
