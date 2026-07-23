import torch, ttnn
from eval.sharding import auto_shard_config, shard_config

# device grid
g = device.compute_with_storage_grid_size()
print("COMPUTE GRID:", g.x, "x", g.y)

# WIDTH loose case geometry
shape = [1, 1, 32, 2048]
mc = auto_shard_config(
    shape, ttnn.TensorMemoryLayout.WIDTH_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
)
print("WIDTH 2048 shard_spec:", mc.shard_spec.shape, "grid:", mc.shard_spec.grid)

# BLOCK loose case geometry
shape2 = [1, 1, 256, 512]
mc2 = auto_shard_config(
    shape2, ttnn.TensorMemoryLayout.BLOCK_SHARDED, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
)
print("BLOCK 256x512 shard_spec:", mc2.shard_spec.shape, "grid:", mc2.shard_spec.grid)

# perf WIDTH geometry (explicit)
mc3 = shard_config(
    [32, 160],
    (8, 4),
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat16,
    device=device,
)
print("WIDTH perf 5120 shard:", mc3.shard_spec.shape, "grid:", mc3.shard_spec.grid)

# Confirm current op rejects WIDTH_SHARDED at validate
from ttnn.operations.rms_norm import rms_norm

x = torch.randn(shape, dtype=torch.bfloat16)
xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
try:
    out = rms_norm(xt)
    print("UNEXPECTED: op ran, out mem:", out.memory_config().memory_layout)
except Exception as e:
    print("REJECTED (expected):", type(e).__name__, str(e)[:120])
