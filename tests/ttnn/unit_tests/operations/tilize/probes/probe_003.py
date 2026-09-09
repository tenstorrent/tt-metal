import torch, ttnn

dev = ttnn.open_device(device_id=0)
try:
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    ss = ttnn.ShardSpec(grid, [32, 64], ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, ss)
    print("memory_layout:", mc.memory_layout)
    print("shard_spec:", mc.shard_spec)
    print("nd_shard_spec:", getattr(mc, "nd_shard_spec", "MISSING"))
    print("is None?", getattr(mc, "nd_shard_spec", None) is None)
finally:
    ttnn.close_device(dev)
