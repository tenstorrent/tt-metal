import torch, ttnn
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as pd
from eval.sharding import auto_shard_config

device = ttnn.open_device(device_id=0)
shape = (128, 8192)
dt = ttnn.float32
mc = auto_shard_config(
    list(shape), ttnn.TensorMemoryLayout.BLOCK_SHARDED, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dt, device=device
)
print("shard", mc.shard_spec.shape, "grid", mc.shard_spec.grid)
x = ttnn.from_torch(
    torch.zeros(shape, dtype=torch.float32), dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc
)
r = ttnn.from_torch(
    torch.zeros(shape, dtype=torch.float32), dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc
)
W = 8192
g = ttnn.from_torch(torch.zeros(1, 1, 1, W, dtype=torch.float32), dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
b = ttnn.from_torch(torch.zeros(1, 1, 1, W, dtype=torch.float32), dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dt, ttnn.ROW_MAJOR_LAYOUT, device, mc)
print("shard l1 bytes each", pd._shard_l1_bytes(x))
desc = pd.create_program_descriptor(
    x, out, weight=g, bias=b, residual=r, epsilon=1e-12, compute_kernel_config=ttnn.ComputeConfigDescriptor()
)
tot = 0
for cb in desc.cbs:
    fd = cb.format_descriptors[0]
    n = cb.total_size // fd.page_size
    print(f"  cb{fd.buffer_index:3d} pages={n:4d} page={fd.page_size:6d} bytes={cb.total_size:8d}")
    tot += cb.total_size
print("TOTAL CB BYTES", tot)
ttnn.close_device(device)
