import torch, ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    shape, G = (1, 1, 16384, 320), 32
    N, _, HW, C = shape
    x = torch.randn(shape, dtype=torch.bfloat16)
    ref = torch.nn.functional.group_norm(x.float().squeeze(1).permute(0, 2, 1), G).permute(0, 2, 1).unsqueeze(1)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y = ttnn.to_torch(groupnorm_sc_N_1_HW_C(tx, G)).float()
    err = (y - ref).squeeze()
    Cg = C // G
    for g in [0, 1, 3, 15, 31]:
        e = err[:, g * Cg : (g + 1) * Cg]
        print(f"PROBE g={g} bias={e.mean():+.5f} std={e.std():.5f}")
    # implied mean/rstd errors per group:
    xg = x.float().squeeze()
    g0 = err[:, 0:Cg]
    print(f"PROBE total rms={err.pow(2).mean().sqrt():.5f} bias_all={err.mean():+.5f}")
finally:
    ttnn.close_device(device)
