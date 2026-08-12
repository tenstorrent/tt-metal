import torch, ttnn

grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
dev = ttnn.open_device(device_id=0)
try:
    for shape, sh in (
        ([4, 128, 128], [2, 64, 64]),
        ([3, 160, 160], [2, 64, 64]),
        ([23, 96, 160], [4, 64, 96]),
        ([5, 4, 160, 160], [2, 3, 64, 96]),
    ):
        mem = ttnn.MemoryConfig(
            buffer_type=ttnn.BufferType.L1,
            nd_shard_spec=ttnn.NdShardSpec(shard_shape=sh, grid=grid, orientation=ttnn.ShardOrientation.ROW_MAJOR),
        )
        rm = ttnn.from_torch(
            torch.rand(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            memory_config=mem,
        )
        tl = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, mem)
        cbr = ttnn.cb_descriptor_from_sharded_tensor(0, rm)
        cbt = ttnn.cb_descriptor_from_sharded_tensor(1, tl)
        import math

        box_rm = math.prod(sh[:-1]) * sh[-1] * 2
        box_tl = (math.prod(sh[:-1]) // 32) * (sh[-1] // 32) * 2048
        print(
            shape,
            sh,
            "rm bank",
            cbr.total_size,
            "box",
            box_rm,
            "ratio",
            cbr.total_size / box_rm,
            "| tile bank",
            cbt.total_size,
            "box",
            box_tl,
            "ratio",
            cbt.total_size / box_tl,
        )
finally:
    ttnn.close_device(dev)
