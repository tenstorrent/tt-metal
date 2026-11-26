import torch, ttnn
from ttnn.operations.g6_typecast import g6_typecast

device = ttnn.open_device(device_id=0)
torch.manual_seed(42)
shape = (1, 1, 256, 256)
x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
out = ttnn.to_torch(g6_typecast(xt, ttnn.bfloat8_b)).to(torch.float64)
host = ttnn.to_torch(ttnn.from_torch(x.to(torch.float32), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).to(
    torch.float64
)
# device-side reference via ttnn bf8b from a device bf16 tensor is not used: compare with host only
a = out.reshape(-1, 16)
h = host.reshape(-1, 16)
ex = x.to(torch.float64).reshape(-1, 16)
gmax = ex.abs().amax(dim=1, keepdim=True)
_, e = torch.frexp(gmax)
step = torch.pow(2.0, (e - 7).to(torch.float64))
bad = (a - h).abs() > step
rows = torch.nonzero(bad.any(dim=1)).reshape(-1)[:6].tolist()
print("n_bad", int(bad.sum()), "groups", int(bad.any(dim=1).sum()))
for r in rows:
    print("group", r, "step", step[r].item())
    print(" x   ", [f"{v:.6g}" for v in ex[r].tolist()])
    print(" dev ", [f"{v:.6g}" for v in a[r].tolist()])
    print(" host", [f"{v:.6g}" for v in h[r].tolist()])
# Also: device vs exact error stats
err_dev = ((a - ex).abs() / step).max().item()
err_host = ((h - ex).abs() / step).max().item()
print("max |dev-exact|/step", err_dev, " max |host-exact|/step", err_host)
ttnn.close_device(device)
