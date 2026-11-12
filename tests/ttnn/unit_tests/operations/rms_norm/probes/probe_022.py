import torch, ttnn, math
from eval.sharding import auto_shard_config
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd

device = ttnn.open_device(device_id=0)
shape = (128, 8192)
mc = auto_shard_config(
    shape, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32, device=device
)
print("MC", mc)
x = ttnn.from_torch(
    torch.randn(shape), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc
)
out = ttnn.allocate_tensor_on_device(x.shape, x.dtype, x.layout, device, mc)
g = pd._Geometry(x, None, out)
v = ttnn.device.get_memory_view(device, ttnn.BufferType.L1)
budget = v.largest_contiguous_bytes_free_per_bank - pd.L1_SAFETY_MARGIN
print(
    "budget",
    budget,
    "free/bank",
    v.largest_contiguous_bytes_free_per_bank,
    "rm_in_place",
    g.rm_in_place,
    "shard_h",
    g.shard_h,
)
for K in (1, 32):
    print("fp K", K, pd._footprint(g, 256, 1, K))
g.rm_in_place = g.x_resident = False
print("fp ls K32", pd._footprint(g, 256, 1, 32))
ttnn.close_device(device)
