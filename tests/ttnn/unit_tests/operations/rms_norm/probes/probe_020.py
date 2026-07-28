import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
shape = (1, 1, 32, 64)
t = torch.ones(*shape, dtype=torch.bfloat16)
ti = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
o = ttnn.to_torch(rms_norm(ti)).float()
print("out[0]", o[0, 0, 0, 0].item())
ttnn.close_device(device)
