import torch, ttnn
from ttnn.operations.tilize.tilize import tilize

dev = ttnn.open_device(device_id=0)
for mem in [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG]:
    for shape in [
        (1, 1, 32, 32),
        (1, 1, 64, 32),
        (1, 1, 32, 96),
        (1, 1, 32, 1056),
        (2, 3, 96, 1056),
        (1, 1, 32, 8224),
        (1, 1, 3200, 32),
    ]:
        x = torch.randint(0, 256, shape, dtype=torch.uint8)
        t = ttnn.from_torch(x, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mem)
        y = ttnn.to_torch(tilize(t))
        print("RES u8", mem.buffer_type, shape, "nbad", (y != x).sum().item())
        xf = torch.randn(shape).to(torch.float8_e4m3fn).to(torch.float32)
        t = ttnn.from_torch(xf, dtype=ttnn.fp8_e4m3, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mem)
        y = ttnn.to_torch(tilize(t)).float()
        a = xf.double().flatten()
        b = y.double().flatten()
        print(
            "RES fp8",
            mem.buffer_type,
            shape,
            f"pcc={torch.corrcoef(torch.stack([a,b]))[0,1].item():.6f} nbad={(a!=b).sum().item()} nbad_nonsub={((a!=b)&(a.abs()>=2**-6)).sum().item()}",
        )
# padded uint8 + fp8 with negative fill
x = torch.randint(0, 256, (1, 1, 50, 50), dtype=torch.uint8)
t = ttnn.from_torch(x, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
y = tilize(t, pad_value=7).cpu().to_torch_with_padded_shape()
print("RES pad u8", torch.equal(y, torch.nn.functional.pad(x, (0, 14, 0, 14), value=7)))
xf = torch.randn(1, 1, 50, 50).to(torch.float8_e4m3fn).to(torch.float32)
t = ttnn.from_torch(xf, dtype=ttnn.fp8_e4m3, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
y = tilize(t, pad_value=-2.0, dtype=ttnn.bfloat16).cpu().to_torch_with_padded_shape().float()
e = torch.nn.functional.pad(xf, (0, 14, 0, 14), value=-2.0)
print("RES pad fp8 nbad_nonsub", ((y != e) & (e.abs() >= 2**-6)).sum().item())
ttnn.close_device(dev)
