import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

dev = device
cases = [(1, 1, 32, 4096), (1, 1, 64, 8192), (1, 1, 64, 12288)]
for shp in cases:
    x = torch.ones(shp)
    ti = ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    out = ttnn.to_torch(rms_norm(ti)).float()
    print(shp, "mean_out=", out.mean().item(), "expected~1.0")
