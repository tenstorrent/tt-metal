import ttnn
import torch
import random

device_id = 0
device = ttnn.open_device(device_id=device_id)

x = torch.Tensor(size=[1, 1, 32, 32]).uniform_(-100, 100).to(torch.bfloat16)
tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

dim = 3
inds_dim = random.randint(0, x.size(dim) - 1)

print(inds_dim)

ttnn.close_device(device)
