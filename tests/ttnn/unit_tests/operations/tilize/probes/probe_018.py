import torch, ttnn
from ttnn.operations.tilize.tilize import tilize

torch.set_printoptions(linewidth=200, precision=4, sci_mode=False)
device = ttnn.open_device(device_id=0)
x = torch.zeros(1, 1, 32, 32)
x[0, 0, 0, :16] = torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 0.5, 0.3, 0.26, 0.1, -1.3, -1.9])
x[0, 0, 1, :16] = 3.0
x[0, 0, 2, :] = torch.arange(32).float()
x = x.bfloat16()
t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
y = ttnn.to_torch(tilize(t, dtype=ttnn.bfloat4_b)).float()
ref = ttnn.to_torch(ttnn.from_torch(x.float(), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()
for r in range(3):
    print("in ", x[0, 0, r, :16].float())
    print("dev", y[0, 0, r, :16])
    print("ref", ref[0, 0, r, :16])
print("in ", x[0, 0, 2, 16:].float())
print("dev", y[0, 0, 2, 16:])
print("ref", ref[0, 0, 2, 16:])
ttnn.close_device(device)
