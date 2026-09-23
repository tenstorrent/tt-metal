import ttnn, torch

dev = ttnn.open_device(device_id=0)
grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
ss = ttnn.ShardSpec(grid, (128, 64), ttnn.ShardOrientation.ROW_MAJOR)
mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, ss)
t = ttnn.from_torch(
    torch.zeros(1, 1, 512, 64, dtype=torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc
)
m = t.memory_config()
print(
    "legacy tensor: layout", m.memory_layout, "shard_spec", m.shard_spec is not None, "nd", m.nd_shard_spec is not None
)
nd = ttnn.NdShardSpec(
    ttnn.Shape([1, 1, 64, 64]),
    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
    ttnn.ShardOrientation.ROW_MAJOR,
)
t2 = ttnn.from_torch(
    torch.zeros(1, 1, 128, 64, dtype=torch.bfloat16),
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=dev,
    memory_config=ttnn.MemoryConfig(ttnn.BufferType.L1, nd),
)
m2 = t2.memory_config()
print(
    "nd tensor: layout", m2.memory_layout, "shard_spec", m2.shard_spec is not None, "nd", m2.nd_shard_spec is not None
)
ttnn.close_device(dev)
