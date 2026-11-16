import torch, ttnn
from ttnn.operations.wq_b_linear import wq_b_linear

device = ttnn.open_device(device_id=0)
torch.manual_seed(0)
x = torch.randn(1, 1, 160, 1536).bfloat16()
w = (torch.randn(1536, 4096) * 0.02).bfloat16()
tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
y = ttnn.to_torch(wq_b_linear(tx, tw)).double()[0, 0]
for m in range(5):
    print("m", m, y[m * 32, 110 * 32 : 110 * 32 + 4].tolist(), y[m * 32 + 1, 110 * 32 : 110 * 32 + 2].tolist())
ttnn.close_device(device)
