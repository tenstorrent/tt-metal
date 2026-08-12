import torch, ttnn, math


def show(name, t):
    mc = t.memory_config()
    print(
        f"{name}: shape={list(t.shape)} page_size={t.buffer_page_size()} aligned={t.buffer_aligned_page_size()} "
        f"num_pages={t.buffer_num_pages()} layout={mc.memory_layout}"
    )


dev = ttnn.open_device(device_id=0)
try:
    crs = lambda a, b: ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*a), ttnn.CoreCoord(*b))})
    ROW = ttnn.ShardOrientation.ROW_MAJOR

    # 1. interleaved RM
    t = ttnn.from_torch(
        torch.randn(1, 1, 64, 512).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev
    )
    show("interleaved [1,1,64,512]", t)

    # 2. width-sharded RM L1  (64,128) on 4 cores, W=512
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(crs((0, 0), (3, 0)), (64, 128), ROW)
    )
    t2 = ttnn.from_torch(
        torch.randn(1, 1, 64, 512).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=mc,
    )
    show("width-sharded (64,128)", t2)

    # 3. nd-sharded RM (2,64,96) on [7,128,128] 2x2
    nd = ttnn.NdShardSpec(ttnn.Shape([2, 64, 96]), crs((0, 0), (1, 1)), ROW)
    mc3 = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd)
    t3 = ttnn.from_torch(
        torch.randn(7, 128, 128).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=mc3,
    )
    show("nd (2,64,96) on [7,128,128]", t3)

    # 4. nd-sharded RM (2,64,64) on [4,128,128] 2x2
    nd4 = ttnn.NdShardSpec(ttnn.Shape([2, 64, 64]), crs((0, 0), (1, 1)), ROW)
    mc4 = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1, nd_shard_spec=nd4)
    t4 = ttnn.from_torch(
        torch.randn(4, 128, 128).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=mc4,
    )
    show("nd (2,64,64) on [4,128,128]", t4)

    # 5. block-sharded RM
    mc5 = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, ttnn.ShardSpec(crs((0, 0), (1, 1)), (64, 64), ROW)
    )
    t5 = ttnn.from_torch(
        torch.randn(1, 1, 128, 128).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=dev,
        memory_config=mc5,
    )
    show("block-sharded (64,64) on [1,1,128,128]", t5)
finally:
    ttnn.close_device(dev)
