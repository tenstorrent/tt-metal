import torch, ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

device = ttnn.open_device(device_id=0)
try:
    x = torch.ones((1, 1, 64, 320), dtype=torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(groupnorm_sc_N_1_HW_C(tx, 32))
    print("A max |y| =", out.abs().max().item(), "(expected 0)")
finally:
    ttnn.close_device(device)
