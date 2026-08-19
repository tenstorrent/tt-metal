import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
try:
    shape = (1, 1, 32, 16384)
    x = torch.ones(shape, dtype=torch.float32)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    o = ttnn.to_torch(rms_norm(tx)).float()[0, 0]
    bad = (o - 1.0).abs() > 0.01
    print("num bad", int(bad.sum()), "of", o.numel())
    idx = bad.nonzero()
    print("first 20 bad idx", idx[:20].tolist())
    print("bad col ranges:", idx[:, 1].min().item(), idx[:, 1].max().item())
    print("bad rows:", sorted(set(idx[:, 0].tolist()))[:40])
    # per-column-block summary
    for c0 in range(0, 16384, 122 * 32):
        seg = bad[:, c0 : c0 + 122 * 32]
        print("chunkstart", c0, "badcount", int(seg.sum()))
finally:
    ttnn.close_device(dev)
