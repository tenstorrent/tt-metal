import torch, ttnn
from ttnn.operations.g6_unsqueeze import g6_unsqueeze
import ttnn.operations.g6_unsqueeze.g6_unsqueeze_program_descriptor as pd_mod

dev = ttnn.open_device(device_id=0)
x = ttnn.from_torch(torch.randn(1, 512), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
o = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 512, 1]), ttnn.float32, ttnn.TILE_LAYOUT, dev, x.memory_config())
pd = pd_mod.create_program_descriptor(x, o)
k = pd.kernels[0]
print("PROBE type", type(pd.kernels), type(k))
ra = pd.kernels[0].runtime_args
c = pd_mod.select_plan(x).core_assignments[0][0]
print("PROBE before", list(ra[c.x][c.y]))
pd.kernels[0].runtime_args[c.x][c.y][0] = 12345
print("PROBE after elem set", list(pd.kernels[0].runtime_args[c.x][c.y]))
pd.kernels[0].runtime_args[c.x][c.y] = [1, 2, 3, 4]
print("PROBE after row set", list(pd.kernels[0].runtime_args[c.x][c.y]))
ttnn.close_device(dev)
