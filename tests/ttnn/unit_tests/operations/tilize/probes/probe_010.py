import torch, ttnn
from ttnn.operations.tilize.tilize import tilize

dev = ttnn.open_device(device_id=0)


def run(x, di, do, **kw):
    t = ttnn.from_torch(x, dtype=di, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    o = tilize(t, dtype=do, **kw)
    return ttnn.to_torch(o) if not kw else o.cpu().to_torch_with_padded_shape()


for shape in [(1, 1, 64, 128), (2, 3, 96, 1056), (1, 1, 32, 32), (1, 1, 64, 64)]:
    x = torch.randint(0, 256, shape, dtype=torch.uint8)
    try:
        y = run(x, ttnn.uint8, ttnn.uint8)
        d = (y.int() - x.int()).abs()
        print(
            "RES u8",
            shape,
            "exact" if d.max() == 0 else f"bad {d.max()} nbad={(d>0).sum().item()}/{d.numel()}",
            y.flatten()[:8].tolist(),
            x.flatten()[:8].tolist(),
        )
    except Exception as e:
        print("RES u8 err", str(e)[:300])
    xf = torch.randn(shape).to(torch.float8_e4m3fn).to(torch.float32)
    for do in [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b]:
        try:
            y = run(xf, ttnn.fp8_e4m3, do).float()
            a = xf.double().flatten()
            b = y.double().flatten()
            print(
                "RES fp8",
                shape,
                do,
                "exact" if torch.equal(a, b) else f"pcc={torch.corrcoef(torch.stack([a,b]))[0,1].item():.5f}",
            )
        except Exception as e:
            print("RES fp8 err", do, str(e)[:300])
# padding with negative int fill
x = torch.randint(-1000, 1000, (1, 1, 50, 50), dtype=torch.int32)
y = run(x, ttnn.int32, ttnn.int32, pad_value=-7)
exp = torch.nn.functional.pad(x, (0, 14, 0, 14), value=-7)
print("RES pad i32", torch.equal(y.int(), exp))
x = torch.randint(0, 60000, (1, 1, 50, 50), dtype=torch.int32)
y = run(x, ttnn.uint16, ttnn.uint16, pad_value=5)
print("RES pad u16", torch.equal(y.int(), torch.nn.functional.pad(x, (0, 14, 0, 14), value=5)))
x = torch.randn((1, 1, 50, 50))
y = run(x, ttnn.float32, ttnn.float32, pad_value=-3.4e38)
print("RES pad f32", torch.equal(y.float(), torch.nn.functional.pad(x, (0, 14, 0, 14), value=-3.4e38)))
# tiny tile int32 / fp32
for di in [ttnn.int32, ttnn.float32, ttnn.uint16]:
    x = (
        torch.randint(-1000, 1000, (1, 1, 64, 128), dtype=torch.int32)
        if di != ttnn.float32
        else torch.randn(1, 1, 64, 128)
    )
    if di == ttnn.uint16:
        x = x.abs()
    t = ttnn.from_torch(x, dtype=di, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
    for th in [16, 1]:
        o = tilize(t, dtype=di, tile=ttnn.Tile([th, 32]))
        print("RES tiny", di, th, torch.equal(ttnn.to_torch(o).to(x.dtype), x))
ttnn.close_device(dev)
