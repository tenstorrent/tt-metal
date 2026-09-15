import torch, ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

device = ttnn.open_device(device_id=0)
try:

    def dev(t, layout):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    for shape, G in (((1, 1, 64, 128), 4), ((1, 1, 64, 128), 1), ((1, 1, 32, 64), 1), ((1, 1, 32, 32), 1)):
        C = shape[-1]
        x = torch.ones(shape, dtype=torch.bfloat16)
        gamma = torch.full((1, 1, 1, C), 2.0, dtype=torch.bfloat16)
        beta = (
            (torch.arange(C, dtype=torch.float32) / 64.0 + 1.0).reshape(1, 1, 1, C).to(torch.bfloat16)
        )  # 1 + c/64, exact
        for glayout, name in ((ttnn.ROW_MAJOR_LAYOUT, "RM"), (ttnn.TILE_LAYOUT, "TILE")):
            out = ttnn.to_torch(
                groupnorm_sc_N_1_HW_C(dev(x, ttnn.TILE_LAYOUT), G, gamma=dev(gamma, glayout), beta=dev(beta, glayout))
            ).float()
            bf = beta.float()[0, 0, 0]
            got0 = out[0, 0, 0]
            bad = [c for c in range(C) if abs(float(got0[c]) - float(bf[c])) > 1e-3]
            # which beta lane did each bad lane receive?
            src = []
            for c in bad[:12]:
                m = torch.nonzero((bf - got0[c]).abs() < 1e-3).flatten().tolist()
                src.append((c, m[:3]))
            rows_uniform = all(torch.equal(out[0, 0, r], out[0, 0, 0]) for r in range(shape[2]))
            print(
                f"--- {shape} G={G} beta {name}: {len(bad)} bad lanes; (lane -> beta lane received) {src}; rows identical: {rows_uniform}"
            )
finally:
    ttnn.close_device(device)
