import torch, ttnn
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    shape, G = (1, 1, 32, 320), 32
    x = torch.randn(shape, dtype=torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ty = groupnorm_sc_N_1_HW_C(tx, G)
    y = ttnn.to_torch(ty).float()
    print("done", flush=True)
finally:
    ttnn.close_device(device)
