import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
try:
    shape = (1, 1, 32, 16384)
    x = torch.ones(shape, dtype=torch.float32)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    for trial in range(3):
        o = ttnn.to_torch(rms_norm(tx)).float()[0, 0]
        bad = (o - 1.0).abs() > 0.01
        tiles = bad.reshape(32, 512, 32).any(0).any(-1).nonzero().flatten().tolist()
        print("trial", trial, "nbad", int(bad.sum()), "bad tiles", tiles)
        if trial == 0:
            t = tiles[0] if tiles else None
            if t is not None:
                print("  tile", t, "row0 vals", o[0, t * 32 : t * 32 + 32].tolist())
finally:
    ttnn.close_device(dev)
