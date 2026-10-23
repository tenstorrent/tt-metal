import torch, ttnn
from eval.sharding import shard_config
from ttnn.operations.rms_norm.rms_norm_program_descriptor import _l1_budget_bytes

device = ttnn.open_device(device_id=0)
print("budget empty", _l1_budget_bytes(device))
view = ttnn._ttnn.device.GetMemoryView(device, ttnn.BufferType.L1)
print("view total", view.total_bytes_per_bank)
mc = shard_config(
    [1024, 128],
    (8, 8),
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat16,
    device=device,
)
t = ttnn.from_torch(
    torch.randn(1, 1, 8192, 1024), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
)
print("addr", t.buffer_address(), "shard bytes", 1024 * 128 * 2)
print("budget after", _l1_budget_bytes(device))
view = ttnn._ttnn.device.GetMemoryView(device, ttnn.BufferType.L1)
for b in view.block_table:
    print("blk", dict(b))
ttnn.close_device(device)
