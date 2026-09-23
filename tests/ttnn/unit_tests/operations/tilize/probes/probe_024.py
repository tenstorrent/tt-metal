import torch, ttnn
from ttnn.operations.tilize import tilize

dev = ttnn.open_device(device_id=0)
x = torch.randint(0, 256, (32, 32), dtype=torch.int32).to(torch.uint8)
t = ttnn.from_torch(x, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
g = ttnn.to_torch(tilize(t))
print(
    "RES",
    g.dtype,
    g.flatten()[:6].tolist(),
    x.flatten()[:6].tolist(),
    g.double().flatten()[:6].tolist(),
    x.double().flatten()[:6].tolist(),
    torch.equal(g, x),
)
e, a = x.double(), g.double()
print("RES", (a - e).abs().max().item())
ttnn.close_device(dev)
