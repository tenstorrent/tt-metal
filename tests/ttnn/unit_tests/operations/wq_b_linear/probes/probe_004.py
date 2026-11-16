import torch, ttnn
from ttnn.operations.wq_b_linear import wq_b_linear

device = ttnn.open_device(device_id=0)
torch.manual_seed(0)
x = torch.randn(1, 1, 160, 1536).bfloat16()
w = (torch.randn(1536, 4096) * 0.02).bfloat16()
tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
ref = torch.matmul(x.double(), ttnn.to_torch(tw).double())[0, 0]
y = ttnn.to_torch(wq_b_linear(tx, tw)).double()[0, 0]
err = (y - ref).abs().reshape(5, 32, 128, 32).amax(dim=(1, 3))
bad = (err > 0.1) | ~torch.isfinite(err)
print("bad tiles (m, n):", [(int(a), int(b)) for a, b in bad.nonzero()][:60])
print("bad columns:", sorted(set(int(b) for _, b in bad.nonzero())))
for n in (0, 109, 110, 111, 127):
    print(n, err[:, n].tolist())
ttnn.close_device(device)
