import torch, ttnn
import ttnn.operations.rms_norm.rms_norm_program_descriptor as desc
from ttnn.operations.rms_norm import rms_norm

desc._FORCE_REGIME = "B"
desc._FORCE_TRANSPORT = 2
desc._FORCE_K = 8
device = ttnn.open_device(device_id=0)
try:
    shp = (1, 1, 32, 2048)
    xo = torch.ones(*shp)
    to = ttnn.from_torch(xo.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    oo = ttnn.to_torch(rms_norm(to)).float()[0, 0, 0]
    Wts = 2048 // 8
    print("all-ones per-shard-first-col:", [round(oo[i * Wts].item(), 3) for i in range(8)])
finally:
    ttnn.close_device(device)
