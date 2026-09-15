import torch, ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

device = ttnn.open_device(device_id=0)
try:

    def dev(t, layout):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    torch.manual_seed(1)
    for shape, G in (((1, 1, 64, 128), 4), ((1, 1, 32, 64), 1)):
        C, HW = shape[-1], shape[-2]
        x = torch.randn(shape).to(torch.bfloat16)
        gamma = torch.randn(1, 1, 1, C).to(torch.bfloat16)
        beta = torch.randn(1, 1, 1, C).to(torch.bfloat16)
        for glayout, name in ((ttnn.ROW_MAJOR_LAYOUT, "RM"), (ttnn.TILE_LAYOUT, "TILE")):
            out = ttnn.to_torch(
                groupnorm_sc_N_1_HW_C(dev(x, ttnn.TILE_LAYOUT), G, gamma=dev(gamma, glayout), beta=dev(beta, glayout))
            ).float()
            xf = x.float()[0, 0]  # (HW, C)
            Cg = C // G
            z = torch.empty_like(xf)
            for g in range(G):
                blk = xf[:, g * Cg : (g + 1) * Cg]
                z[:, g * Cg : (g + 1) * Cg] = (blk - blk.mean()) * torch.rsqrt(blk.var(unbiased=False) + 1e-5)
            y = out[0, 0]
            # per-lane least squares y = z*g_eff + b_eff
            zm = z.mean(0)
            ym = y.mean(0)
            g_eff = ((z - zm) * (y - ym)).sum(0) / ((z - zm) ** 2).sum(0)
            b_eff = ym - g_eff * zm
            gerr = (g_eff - gamma.float()[0, 0, 0]).abs().max().item()
            berr = (b_eff - beta.float()[0, 0, 0]).abs().max().item()
            print(f"--- {shape} G={G} affine {name}: max|gamma_eff-gamma|={gerr:.4f} max|beta_eff-beta|={berr:.4f}")
finally:
    ttnn.close_device(device)
