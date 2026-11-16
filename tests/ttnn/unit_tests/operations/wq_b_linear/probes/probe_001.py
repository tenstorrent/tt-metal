import torch, ttnn
from ttnn.operations.wq_b_linear import wq_b_linear

dev = ttnn.open_device(device_id=0)
g = dev.compute_with_storage_grid_size()
print("grid", g.x, g.y)
print("split", ttnn.split_work_to_cores(g, 128, row_wise=True)[0::4])
torch.manual_seed(0)
x = torch.randn(1, 1, 160, 1536).bfloat16()
w = (torch.randn(1536, 4096) * 0.02).bfloat16()
tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=dev)
ref = torch.matmul(x.double(), ttnn.to_torch(tw).double())
y = ttnn.to_torch(wq_b_linear(tx, tw)).double()
a = (y - y.mean()).flatten()
b = (ref - ref.mean()).flatten()
print("pcc", float((a @ b) / (a.norm() * b.norm())))
ttnn.close_device(dev)
