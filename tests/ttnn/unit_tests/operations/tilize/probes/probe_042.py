import torch, ttnn
from ttnn.operations.tilize import tilize

dev = ttnn.open_device(device_id=0)
try:
    t = torch.randn([1, 1, 256, 256]).to(torch.float32)
    src = ttnn.from_torch(t, dtype=ttnn.float32, device=dev, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile([32, 32]))
    got = ttnn.to_torch(tilize(src, tile=ttnn.Tile([8, 32]), dtype=ttnn.bfloat16))
    ref = t.to(torch.bfloat16)
    print("maxerr", (got.float() - ref.float()).abs().max().item())
    print("OK exact-vs-bf16-rounding" if torch.equal(got, ref) else "BAD")
    # and the reverse direction, plus bf16 -> bf8b
    t2 = torch.randn([1, 1, 256, 256]).to(torch.bfloat16)
    s2 = ttnn.from_torch(t2, dtype=ttnn.bfloat16, device=dev, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile([32, 32]))
    g2 = ttnn.to_torch(tilize(s2, tile=ttnn.Tile([8, 32]), dtype=ttnn.bfloat8_b))
    print("OK bf8b" if (g2.float() - t2.float()).abs().max().item() < 0.06 else "BAD bf8b")
finally:
    ttnn.close_device(dev)
