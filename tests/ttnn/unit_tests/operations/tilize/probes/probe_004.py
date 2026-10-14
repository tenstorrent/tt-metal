import torch, ttnn

dev = ttnn.open_device(device_id=0)
g = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
a = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(g, (64, 64), ttnn.ShardOrientation.ROW_MAJOR),
)
b = ttnn.MemoryConfig(
    ttnn.BufferType.L1, ttnn.NdShardSpec(ttnn.Shape([1, 1, 64, 64]), g, ttnn.ShardOrientation.ROW_MAJOR)
)
for m in (a, b):
    t = ttnn.from_torch(
        torch.randn(1, 1, 128, 64).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=m,
    )
    mc = t.memory_config()
    print("RES", mc.memory_layout, mc.shard_spec is not None, mc.nd_shard_spec is not None)
ttnn.close_device(dev)
