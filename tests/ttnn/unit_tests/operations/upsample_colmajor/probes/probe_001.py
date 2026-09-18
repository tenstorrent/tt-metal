import torch, ttnn

dev = ttnn.open_device(device_id=0, l1_small_size=32768)
try:
    for H, C in ((64, 640), (32, 1280)):
        x = torch.randn(1, H, H, C).bfloat16()
        HW = H * H
        tiles = -(-HW // 32)
        cols = 11
        per = -(-tiles // cols)
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 9))})
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [per * 32, C // 10], ttnn.ShardOrientation.COL_MAJOR),
        )
        t = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        t = ttnn.to_memory_config(t, mc)
        print("PROBE input", t.shape, t.memory_config().shard_spec)
        y = ttnn.upsample(t, (2, 2))
        print("PROBE output", y.shape, y.memory_config().shard_spec)
        ref = torch.nn.functional.interpolate(x.permute(0, 3, 1, 2).float(), scale_factor=2, mode="nearest").permute(
            0, 2, 3, 1
        )
        out = ttnn.to_torch(y).float()
        print("PROBE maxdiff", (out - ref).abs().max().item(), "H C", H, C)
finally:
    ttnn.close_device(dev)
