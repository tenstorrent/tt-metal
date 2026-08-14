import torch, ttnn, time, os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
from ttnn.operations.tilize import tilize

dev = ttnn.open_device(device_id=0)
try:
    ok = True
    for in_h, out_h, shape, dt in [
        (32, 8, [1, 1, 1024, 1024], ttnn.bfloat16),
        (32, 16, [1, 1, 1024, 1024], ttnn.bfloat16),
        (32, 4, [1, 1, 256, 256], ttnn.bfloat16),
        (1, 32, [1, 1, 256, 256], ttnn.bfloat16),
        (8, 32, [1, 1, 256, 256], ttnn.bfloat16),
        (32, 8, [1, 1, 256, 256], ttnn.float32),
        (32, 8, [1, 1, 256, 256], ttnn.uint8),
        (32, 16, [1, 1, 256, 256], ttnn.uint32),
    ]:
        if dt == ttnn.uint8:
            t = torch.randint(0, 200, shape, dtype=torch.uint8)
        elif dt in (ttnn.uint32,):
            t = torch.randint(0, 10000, shape, dtype=torch.int32)
        else:
            t = torch.randn(shape).to(torch.bfloat16 if dt == ttnn.bfloat16 else torch.float32)
        src = ttnn.from_torch(t, dtype=dt, device=dev, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile([in_h, 32]))
        out = tilize(src, tile=ttnn.Tile([out_h, 32]))
        got = ttnn.to_torch(out)
        eq = torch.equal(got.to(t.dtype), t)
        ok &= eq
        print(f"retile {in_h}->{out_h} {shape} {dt}: exact={eq}")
    print("ALL EXACT" if ok else "FAILURE")
finally:
    ttnn.close_device(dev)
