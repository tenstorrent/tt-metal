import torch, ttnn
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
shape = (13, 777, 1023)
mc = auto_shard_config(
    list(shape), ttnn.TensorMemoryLayout.WIDTH_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
)
print("PROBE shard", mc.shard_spec)
x = ttnn.from_torch(torch.randn(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
out = ttnn.allocate_tensor_on_device(x.shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
gm = ttnn.from_torch(torch.randn(1, 1, 1, 1023), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
g = pd._Geometry(x, gm, out, False)
work, _ = pd._plan_sharded(g)
b = pd._cb_budget(device)
print("PROBE budget", b, "G", g.group_size)
for r in (1, 2):
    print("PROBE r", r, "F", pd._footprint(g, 1, r, 1))
ttnn.close_device(device)
