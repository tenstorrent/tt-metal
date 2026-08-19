import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
try:
    shape = (1, 1, 32, 64)
    x = torch.ones(shape, dtype=torch.float32)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    g = torch.arange(1, shape[-1] + 1, dtype=torch.float32).reshape(1, 1, 1, shape[-1])
    for glayout, name in [(ttnn.TILE_LAYOUT, "TILE"), (ttnn.ROW_MAJOR_LAYOUT, "RM")]:
        tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=glayout, device=dev)
        out = ttnn.to_torch(rms_norm(tx, gamma=tg)).float()
        print(name, "row0[:8]", out[0, 0, 0, :8].tolist())
        print(name, "row0[-4:]", out[0, 0, 0, -4:].tolist())
        print(name, "row5[:8]", out[0, 0, 5, :8].tolist())
        print(name, "has_nan", bool(torch.isnan(out).any()))
finally:
    ttnn.close_device(dev)
