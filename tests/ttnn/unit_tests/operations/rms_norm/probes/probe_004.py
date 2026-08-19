import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
try:
    for shape, dt in [((1, 1, 32, 4096), ttnn.float32), ((1, 1, 32, 16384), ttnn.bfloat16)]:
        W = shape[-1]
        x = torch.ones(shape, dtype=torch.float32)
        tx = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        o = ttnn.to_torch(rms_norm(tx)).float()
        print(
            shape,
            dt,
            "ONES no-gamma: min",
            o.min().item(),
            "max",
            o.max().item(),
            "first8",
            o[0, 0, 0, :8].tolist(),
            "at2000",
            o[0, 0, 0, 2000:2004].tolist(),
            "last4",
            o[0, 0, 0, -4:].tolist(),
        )
finally:
    ttnn.close_device(dev)
