import itertools, torch, ttnn
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
col = 110
cs = slice(col * 32, col * 32 + 32)
yc = y[:, cs]


def blk(xb, wb):
    return xd[:, xb * 96 : (xb + 1) * 96] @ wq[wb * 96 : (wb + 1) * 96, cs]


cands = {}
for xbs in itertools.product(range(16), repeat=1):
    for wb in range(16):
        cands[f"x{xbs}w{wb}"] = blk(xbs[0], wb)
# sums of 3 consecutive pairs with x offset d
for d in range(-4, 12):
    tot = 0
    ok = True
    for i in range(3):
        xb = 2 + i + d
        if not 0 <= xb < 16:
            ok = False
            break
        tot = tot + blk(xb, 2 + i)
    if ok:
        cands[f"slice1_xshift{d}"] = tot
for i in range(3):
    for j in range(3):
        cands[f"two_{i}{j}"] = blk(2 + i, 2 + i) + blk(2 + j, 2 + j) if i != j else None
cands = {k: v for k, v in cands.items() if v is not None}
for m in range(5):
    rows = slice(m * 32, m * 32 + 32)
    best = sorted((float((yc[rows] - v[rows]).abs().max()), k) for k, v in cands.items())[:3]
    print("mtile", m, best)
ttnn.close_device(device)
