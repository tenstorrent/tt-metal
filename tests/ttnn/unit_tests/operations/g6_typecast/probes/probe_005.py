import torch, ttnn
from ttnn.operations.g6_typecast import g6_typecast

d = ttnn.open_device(device_id=0)
torch.manual_seed(42)
shape = (1, 1, 31, 256)
x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
ti = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d)
dev = ttnn.to_torch(g6_typecast(ti, ttnn.bfloat8_b)).to(torch.float64).reshape(-1, 16)
host = (
    ttnn.to_torch(ttnn.from_torch(x.to(torch.float32), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT))
    .to(torch.float64)
    .reshape(-1, 16)
)
ex = x.to(torch.float64).reshape(-1, 16)
gmax = ex.abs().amax(dim=1, keepdim=True)
_, e = torch.frexp(gmax)
step = torch.pow(2.0, (e - 7).to(torch.float64))
bad = (dev - host).abs() > step
rows = torch.nonzero(bad.any(dim=1)).reshape(-1).tolist()
print("bad groups", len(rows))
for r in rows[:3]:
    print("group", r, "gmax", gmax[r].item(), "step", step[r].item())
    print(" exact", ex[r].tolist())
    print(" host ", host[r].tolist())
    print(" dev  ", dev[r].tolist())
    # how do host / dev deviate from exact
    print(" host-ex/step", ((host[r] - ex[r]) / step[r]).tolist())
    print(" dev-ex/step ", ((dev[r] - ex[r]) / step[r]).tolist())
ttnn.close_device(d)
