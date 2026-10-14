import torch, ttnn
from ttnn.operations.tilize.tilize import tilize

dev = ttnn.open_device(device_id=0)
T = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
    ttnn.uint32: torch.int32,
    ttnn.int32: torch.int32,
    ttnn.uint16: torch.int32,
    ttnn.uint8: torch.int32,
    ttnn.fp8_e4m3: torch.float32,
}
pairs = [
    (ttnn.float32, ttnn.float32),
    (ttnn.float32, ttnn.bfloat16),
    (ttnn.float32, ttnn.bfloat8_b),
    (ttnn.float32, ttnn.bfloat4_b),
    (ttnn.bfloat16, ttnn.float32),
    (ttnn.bfloat16, ttnn.bfloat8_b),
    (ttnn.bfloat16, ttnn.bfloat4_b),
    (ttnn.fp8_e4m3, ttnn.float32),
    (ttnn.fp8_e4m3, ttnn.bfloat16),
    (ttnn.fp8_e4m3, ttnn.bfloat8_b),
    (ttnn.fp8_e4m3, ttnn.bfloat4_b),
    (ttnn.uint32, ttnn.uint32),
    (ttnn.int32, ttnn.int32),
    (ttnn.uint32, ttnn.int32),
    (ttnn.int32, ttnn.uint32),
    (ttnn.uint16, ttnn.uint16),
    (ttnn.uint8, ttnn.uint8),
]
for shape in [(1, 1, 64, 128), (2, 3, 96, 1056)]:
    for di, do in pairs:
        try:
            if T[di].is_floating_point:
                x = (torch.randn(shape) * (1e30 if di == ttnn.float32 else 1)).to(T[di])
            elif di == ttnn.int32:
                x = torch.randint(-100000, 100000, shape, dtype=torch.int32)
            elif di == ttnn.uint8:
                x = torch.randint(0, 255, shape, dtype=torch.int32)
            else:
                x = torch.randint(0, 60000, shape, dtype=torch.int32)
            t = ttnn.from_torch(x, dtype=di, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            ref = ttnn.to_torch(t)
            o = tilize(t, dtype=do)
            y = ttnn.to_torch(o)
            exact = torch.equal(
                y.to(torch.float64) if y.is_floating_point() else y,
                ref.to(torch.float64) if y.is_floating_point() else ref.to(y.dtype),
            )
            a = ref.double().flatten()
            b = y.double().flatten()
            pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item() if a.std() > 0 else 0
            print(
                "RES",
                shape,
                di,
                do,
                o.layout,
                "exact" if exact else f"pcc={pcc:.5f} maxdiff={(a-b).abs().max().item():.3g}",
            )
        except Exception as e:
            print("RES", shape, di, do, "Error", str(e)[:200])
ttnn.close_device(dev)
