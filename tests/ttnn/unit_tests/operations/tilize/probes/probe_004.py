import torch, ttnn

dev = ttnn.open_device(device_id=0)
try:
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    t = torch.randn(1, 1, 512, 64)
    # legacy 2D height sharded
    ss = ttnn.ShardSpec(grid, [256, 64], ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, ss)
    tt = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc)
    live = tt.memory_config()
    print("PROBE legacy: memory_layout", live.memory_layout)
    print("PROBE legacy: shard_spec is None?", live.shard_spec is None)
    print("PROBE legacy: nd_shard_spec is None?", getattr(live, "nd_shard_spec", None) is None)
    # nd
    nd = ttnn.NdShardSpec(ttnn.Shape([256, 64]), grid, ttnn.ShardOrientation.ROW_MAJOR)
    mc2 = ttnn.MemoryConfig(ttnn.BufferType.L1, nd)
    tt2 = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc2)
    live2 = tt2.memory_config()
    print("PROBE nd: memory_layout", live2.memory_layout)
    print("PROBE nd: shard_spec is None?", live2.shard_spec is None)
    print("PROBE nd: nd_shard_spec is None?", getattr(live2, "nd_shard_spec", None) is None)
finally:
    ttnn.close_device(dev)
