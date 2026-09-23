import ttnn, torch

dev = ttnn.open_device(device_id=0)
grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
ss = ttnn.ShardSpec(grid, (128, 64), ttnn.ShardOrientation.ROW_MAJOR)
mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, ss)
t = ttnn.from_torch(
    torch.zeros(1, 1, 512, 64, dtype=torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc
)
print("legacy json", t.memory_config().to_json())
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
print("nd json", t2.memory_config().to_json())
ttnn.close_device(dev)
