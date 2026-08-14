import torch, ttnn
from ttnn.operations.tilize import tilize

dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    t = torch.randn([1, 1, 256, 256]).to(torch.float32)
    src = ttnn.from_torch(t, dtype=ttnn.float32, device=dev, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile([32, 32]))
    got = ttnn.to_torch(tilize(src, tile=ttnn.Tile([8, 32]), dtype=ttnn.bfloat16))
    ref = t.to(torch.bfloat16)
    d = (got.float() - ref.float()).abs()
    print("RESULT maxerr", d.max().item(), "ndiff", int((d > 0).sum()), "of", d.numel())
finally:
    ttnn.close_device(dev)
