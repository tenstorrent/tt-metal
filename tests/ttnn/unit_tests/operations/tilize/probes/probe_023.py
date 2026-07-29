import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
    nd = ttnn.MemoryConfig(
        ttnn.BufferType.L1, ttnn.NdShardSpec(ttnn.Shape([1, 1, 64, 64]), grid, ttnn.ShardOrientation.ROW_MAJOR)
    )
    print("layout:", nd.memory_layout, "shard_spec:", nd.shard_spec, "nd_shard_spec:", nd.nd_shard_spec)
    print("is_sharded:", nd.is_sharded())
finally:
    ttnn.close_device(device)
