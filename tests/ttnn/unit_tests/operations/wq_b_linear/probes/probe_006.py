import torch, ttnn
from ttnn.operations.wq_b_linear import wq_b_linear

device = ttnn.open_device(device_id=0)
torch.manual_seed(0)
x = torch.randn(1, 1, 160, 1536).bfloat16()
w = (torch.randn(1536, 4096) * 0.02).bfloat16()
tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
wq = ttnn.to_torch(tw).double()
xd = x.double()[0, 0]
y = ttnn.to_torch(wq_b_linear(tx, tw)).double()[0, 0]
S, nkb, kbt = 6, 16, 3
for col in (110, 111, 127):
    cs = slice(col * 32, col * 32 + 32)
    yc = y[:, cs]
    parts = []
    for s in range(S):
        k0 = (s * nkb // S) * kbt * 32
        k1 = ((s + 1) * nkb // S) * kbt * 32
        parts.append(xd[:, k0:k1] @ wq[k0:k1, cs])
    full = sum(parts)
    print(
        col,
        "err vs full",
        float((yc - full).abs().max()),
        "vs p0",
        float((yc - parts[0]).abs().max()),
        "vs p1",
        float((yc - parts[1]).abs().max()),
        "y absmax",
        float(yc.abs().max()),
    )
    print("   per-mtile err vs p0", [float((yc - parts[0])[m * 32 : (m + 1) * 32].abs().max()) for m in range(5)])
ttnn.close_device(device)
