import torch, ttnn
from ttnn.operations.g6_to_layout import g6_to_layout

device = ttnn.open_device(device_id=0)
shape = (1, 1, 32, 256)
torch.manual_seed(460)
base = torch.randn(shape)
k = torch.randint(-8, 9, (*shape[:-1], shape[-1] // 16))
x = (base * torch.pow(2.0, k).repeat_interleave(16, dim=-1)).to(torch.bfloat16)
tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
out = g6_to_layout(tt, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
act = ttnn.to_torch(out).reshape(shape).float()
host = ttnn.to_torch(ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).reshape(shape).float()
# Host tilize path on device for comparison
dev_ref = (
    ttnn.to_torch(ttnn.to_layout(tt, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)).reshape(shape).float() if False else None
)
v = host.view(-1, 16)
e = (
    ((host.to(torch.float32).view(torch.int32) >> 23) & 0xFF)
    .reshape(*shape[:-1], 16, 16)
    .amax(-1)
    .repeat_interleave(16, -1)
)
step = torch.pow(2.0, (e - 133).double())
bad = ~((act.double() - host.double()).abs() <= step)
idx = bad.nonzero()
print("bad", idx.tolist())
for i in idx.tolist()[:6]:
    r, c = i[2], i[3]
    g0 = (c // 16) * 16
    print(
        "pos",
        r,
        c,
        "x",
        x[0, 0, r, c].item(),
        "host",
        host[0, 0, r, c].item(),
        "dev",
        act[0, 0, r, c].item(),
        "step",
        step[0, 0, r, c].item(),
    )
    print("  group x   ", [round(v, 4) for v in x[0, 0, r, g0 : g0 + 16].float().tolist()])
    print("  group host", [round(v, 4) for v in host[0, 0, r, g0 : g0 + 16].tolist()])
    print("  group dev ", [round(v, 4) for v in act[0, 0, r, g0 : g0 + 16].tolist()])
ttnn.close_device(device)
