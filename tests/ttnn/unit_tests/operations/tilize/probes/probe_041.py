import torch, ttnn
from ttnn.operations.tilize import tilize

dev = ttnn.open_device(device_id=0)
try:
    ok = True
    cases = [
        # (shape, in_tile_h, out_tile_h, in_dtype, out_dtype)
        ([1, 1, 256, 256], 32, 8, ttnn.bfloat16, ttnn.float32),
        ([1, 1, 256, 256], 32, 16, ttnn.bfloat16, ttnn.float32),
        ([1, 1, 256, 256], 32, 8, ttnn.float32, ttnn.bfloat16),
        ([1, 1, 256, 256], 8, 32, ttnn.bfloat16, ttnn.float32),
        ([1, 1, 256, 256], 32, 1, ttnn.bfloat16, ttnn.float32),  # the carve-out + cast
        ([1, 1, 256, 256], 32, 1, ttnn.bfloat16, None),  # the carve-out
        ([1, 1, 256, 256], 32, 2, ttnn.bfloat16, None),
        ([1, 1, 256, 256], 32, 4, ttnn.uint8, None),
        ([1, 1, 256, 256], 32, 1, ttnn.uint8, None),
        ([1, 1, 256, 256], 1, 32, ttnn.uint8, None),
        ([1, 1, 256, 256], 32, 8, ttnn.uint32, None),
        ([2, 3, 128, 128], 32, 8, ttnn.bfloat16, None),
    ]
    for shape, ith, oth, idt, odt in cases:
        if idt == ttnn.uint8:
            t = torch.randint(0, 200, shape, dtype=torch.uint8)
        elif idt == ttnn.uint32:
            t = torch.randint(0, 10000, shape, dtype=torch.int32)
        else:
            t = torch.randn(shape).to(torch.bfloat16 if idt == ttnn.bfloat16 else torch.float32)
        src = ttnn.from_torch(t, dtype=idt, device=dev, layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile([ith, 32]))
        kw = dict(tile=ttnn.Tile([oth, 32]))
        if odt is not None:
            kw["dtype"] = odt
        got = ttnn.to_torch(tilize(src, **kw))
        ref = t.to(got.dtype)
        same = torch.equal(got, ref) if odt in (None,) else torch.allclose(got.float(), t.float(), atol=1e-2)
        print(f"{'OK ' if same else 'BAD'} {shape} {ith}->{oth} {idt}->{odt}")
        ok = ok and same
    print("ALL EXACT" if ok else "MISMATCH")
finally:
    ttnn.close_device(dev)
