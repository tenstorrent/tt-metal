import torch, ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C_b.groupnorm_sc_N_1_HW_C_b_program_descriptor import plan_or_raise, formats_of

device = ttnn.open_device(device_id=0)
for name, shape in [
    ("vae_dram_65536x512", (1, 1, 65536, 512)),
    ("vae_dram_262144x256", (1, 1, 262144, 256)),
    ("refiner_dram_16384x1152", (1, 1, 16384, 1152)),
    ("mochi_4x6360x768", (4, 1, 6360, 768)),
]:
    C = shape[-1]
    for dt in (ttnn.bfloat16, ttnn.bfloat8_b):
        x = ttnn.from_torch(torch.zeros(shape, dtype=torch.bfloat16), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)
        g = ttnn.from_torch(
            torch.zeros(1, 1, 1, C, dtype=torch.bfloat16), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device
        )
        p = plan_or_raise(x, 32, has_gamma=True, has_beta=True, fmt=formats_of(x, g, g))
        print(
            f"PLAN {name} {dt}: K={p.K_alloc} B_max={p.B_max} T={p.T} W={p.W} footprint={p.footprint} cores={len(p.work)}"
        )
        x.deallocate()
        g.deallocate()
ttnn.close_device(device)
