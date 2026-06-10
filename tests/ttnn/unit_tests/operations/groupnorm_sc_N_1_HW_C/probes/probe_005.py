import torch, ttnn

device = ttnn.open_device(device_id=0)
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

x = torch.ones(1, 1, 64, 17, dtype=torch.bfloat16)
tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
out = ttnn.to_torch(groupnorm_sc_N_1_HW_C(tt_x, 1)).to(torch.float32)
print("probe OUT[0,0,0,:5] =", out[0, 0, 0, :5].tolist(), "max abs:", out.abs().max().item())
ttnn.close_device(device)
