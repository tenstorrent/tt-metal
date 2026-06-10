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
    Cg = C // G
    r = ref.squeeze()
    yy = y.squeeze()
    for g in [0, 3, 15, 31]:
        rg = r[:, g * Cg : (g + 1) * Cg].flatten()
        yg = yy[:, g * Cg : (g + 1) * Cg].flatten()
        slope = (yg * rg).sum() / (rg * rg).sum()
        resid = (yg - slope * rg).std()
        print(f"PROBE g={g} slope={slope:.5f} resid_std={resid:.5f}")
finally:
    ttnn.close_device(device)
