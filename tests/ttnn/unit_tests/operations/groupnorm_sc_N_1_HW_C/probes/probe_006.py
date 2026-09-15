import torch, ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

device = ttnn.open_device(device_id=0)
try:

    def dev(t, layout):
        return ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    shape, G = (1, 1, 32, 32), 1
    C = shape[-1]
    x = torch.ones(shape, dtype=torch.bfloat16)
    gamma = torch.full((1, 1, 1, C), 2.0, dtype=torch.bfloat16)
    beta = (torch.arange(C, dtype=torch.float32) / 64.0 + 1.0).reshape(1, 1, 1, C).to(torch.bfloat16)
    out = ttnn.to_torch(
        groupnorm_sc_N_1_HW_C(
            dev(x, ttnn.TILE_LAYOUT), G, gamma=dev(gamma, ttnn.ROW_MAJOR_LAYOUT), beta=dev(beta, ttnn.ROW_MAJOR_LAYOUT)
        )
    ).float()
    print(
        "--- out r0 l0-8:",
        [round(float(v), 3) for v in out[0, 0, 0, :8]],
        " l16-24:",
        [round(float(v), 3) for v in out[0, 0, 0, 16:24]],
        " r16 l0-8:",
        [round(float(v), 3) for v in out[0, 0, 16, :8]],
    )
    print(
        "--- exp   l0-8:",
        [round(float(v), 3) for v in beta.float()[0, 0, 0, :8]],
        " l16-24:",
        [round(float(v), 3) for v in beta.float()[0, 0, 0, 16:24]],
    )
finally:
    ttnn.close_device(device)
