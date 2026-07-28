import torch, ttnn, math
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
for W in (64, 128, 256):
    shape = (1, 1, 32, W)
    t = torch.ones(*shape, dtype=torch.bfloat16)
    ti = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    o = ttnn.to_torch(rms_norm(ti)).float()
    v = o[0, 0, 0, 0].item()
    print(f"W={W} Wt={W//32}  out[0]={v}  implied mean(x^2)={1.0/(v*v)}")
ttnn.close_device(device)
