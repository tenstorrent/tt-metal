import torch, ttnn

grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
nd = ttnn.NdShardSpec(ttnn.Shape([256, 64]), grid, ttnn.ShardOrientation.ROW_MAJOR)
mc2 = ttnn.MemoryConfig(ttnn.BufferType.L1, nd)
print("PROBE fresh nd: memory_layout", mc2.memory_layout)
print("PROBE fresh nd: shard_spec is None?", mc2.shard_spec is None)
print("PROBE fresh nd: nd_shard_spec is None?", getattr(mc2, "nd_shard_spec", None) is None)
