import torch, ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

device = ttnn.open_device(device_id=0)
try:

    def dev(t, layout):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    for shape, G in (((1, 1, 32, 32), 1), ((1, 1, 64, 128), 4)):
        C = shape[-1]
        x = torch.ones(shape, dtype=torch.bfloat16)
        gamma = torch.full((1, 1, 1, C), 2.0, dtype=torch.bfloat16)
        beta = (torch.arange(C, dtype=torch.float32) / 64.0 + 1.0).reshape(1, 1, 1, C).to(torch.bfloat16)
        for glayout, name in ((ttnn.ROW_MAJOR_LAYOUT, "RM"), (ttnn.TILE_LAYOUT, "TILE")):
            out = ttnn.to_torch(
                groupnorm_sc_N_1_HW_C(dev(x, ttnn.TILE_LAYOUT), G, gamma=dev(gamma, glayout), beta=dev(beta, glayout))
            ).float()
            bf = beta.float()[0, 0, 0]
            for r in (0, 1, 15, 16, 31):
                got = out[0, 0, r]
                mapping = []
                for c in range(min(C, 36)):
                    m = torch.nonzero((bf - got[c]).abs() < 1e-3).flatten().tolist()
                    mapping.append(m[0] if m else "?")
                print(f"--- {shape} G={G} {name} row {r}: lane->beta lane {mapping}")
finally:
    ttnn.close_device(device)
