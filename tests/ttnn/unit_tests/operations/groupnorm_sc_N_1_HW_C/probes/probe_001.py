import torch, ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

device = ttnn.open_device(device_id=0)
try:

    def dev(t, layout):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    for trial in range(2):
        # gamma[c] = c+1 (exact in bf16), one group -> z identical per row; y[:, c] / z = gamma_eff[c]
        shape, G = (1, 1, 32, 64), 1
        C = shape[-1]
        torch.manual_seed(0)
        x = torch.randn(shape).to(torch.bfloat16)
        gamma = torch.arange(1, C + 1, dtype=torch.float32).reshape(1, 1, 1, C).to(torch.bfloat16)
        for glayout, name in ((ttnn.ROW_MAJOR_LAYOUT, "RM"), (ttnn.TILE_LAYOUT, "TILE")):
            out = ttnn.to_torch(groupnorm_sc_N_1_HW_C(dev(x, ttnn.TILE_LAYOUT), G, gamma=dev(gamma, glayout))).float()
            xf = x.float()
            z = (xf - xf.mean()) * torch.rsqrt(xf.var(unbiased=False) + 1e-5)
            # effective gamma per lane: least squares over rows
            zz = z[0, 0]
            yy = out[0, 0]
            geff = (zz * yy).sum(0) / (zz * zz).sum(0)
            bad = [(c, round(float(geff[c]), 2)) for c in range(C) if abs(float(geff[c]) - (c + 1)) > 0.5]
            print(f"--- trial {trial} gamma layout {name}: wrong lanes (lane, gamma_eff): {bad}")
    # beta-only style check: all ones input -> y = beta
    shape, G = (1, 1, 64, 128), 4
    C = shape[-1]
    x = torch.ones(shape, dtype=torch.bfloat16)
    gamma = torch.full((1, 1, 1, C), 2.0, dtype=torch.bfloat16)
    beta = (torch.arange(C, dtype=torch.float32) / 64.0).reshape(1, 1, 1, C).to(torch.bfloat16)
    for glayout, name in ((ttnn.ROW_MAJOR_LAYOUT, "RM"), (ttnn.TILE_LAYOUT, "TILE")):
        out = ttnn.to_torch(
            groupnorm_sc_N_1_HW_C(dev(x, ttnn.TILE_LAYOUT), G, gamma=dev(gamma, glayout), beta=dev(beta, glayout))
        ).float()
        exp = beta.float().expand(shape)
        d = (out - exp).abs()
        bad_cols = torch.nonzero(d.amax(dim=(0, 1, 2)) > 0.01).flatten().tolist()
        print(
            f"--- beta {name}: bad cols {bad_cols}; sample (col, got, exp): {[(c, round(float(out[0,0,0,c]),3), round(float(exp[0,0,0,c]),3)) for c in bad_cols[:8]]}"
        )
        # is the error uniform down the rows?
        if bad_cols:
            c = bad_cols[0]
            print(
                f"--- beta {name}: col {c} distinct row values: {sorted(set(round(float(v),3) for v in out[0,0,:,c]))[:6]}"
            )
finally:
    ttnn.close_device(device)
