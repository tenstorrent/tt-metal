import torch, ttnn
from ttnn.operations.wq_b_linear import wq_b_linear
import ttnn.operations.wq_b_linear.wq_b_linear_program_descriptor as pd

pd.SLICE_EARLY_K_BLOCKS = 0
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
col = 110
cs = slice(col * 32, col * 32 + 32)
yc = y[:, cs]
b = [s * nkb // S * kbt * 32 for s in range(S + 1)]
cands = {}
for s in range(S):
    cands[f"p{s}"] = xd[:, b[s] : b[s + 1]] @ wq[b[s] : b[s + 1], cs]
    L = b[s + 1] - b[s]
    cands[f"x0_w{s}"] = xd[:, 0:L] @ wq[b[s] : b[s + 1], cs]
    for off in range(1, 16):
        k0 = off * 96
        if k0 + L <= 1536:
            cands[f"xkb{off}_w{s}"] = xd[:, k0 : k0 + L] @ wq[b[s] : b[s + 1], cs]
for m in range(5):
    rows = slice(m * 32, m * 32 + 32)
    best = sorted((float((yc[rows] - v[rows]).abs().max()), k) for k, v in cands.items())[:3]
    print("mtile", m, best)
ttnn.close_device(device)
