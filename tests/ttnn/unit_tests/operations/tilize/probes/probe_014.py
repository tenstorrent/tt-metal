import torch, ttnn


def info(name, t):
    mc = t.memory_config()
    print(f"=== {name} ===")
    print("  memory_layout:", mc.memory_layout, "buffer_type:", mc.buffer_type)
    print("  is_sharded:", mc.is_sharded())
    try:
        print("  buffer_page_size:", t.buffer_page_size(), "buffer_num_pages:", t.buffer_num_pages())
    except Exception as e:
        print("  page/num err:", e)
    try:
        ss = mc.nd_shard_spec if mc.memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED else mc.shard_spec
        print(
            "  shard_shape:",
            (ss.shard_shape if mc.memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED else ss.shape),
            "grid num_cores:",
            ss.grid.num_cores(),
            "orient:",
            ss.orientation,
        )
    except Exception as e:
        print("  shard err:", e)


device = ttnn.open_device(device_id=0)
try:
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})  # 4 cores
    # nd [4,128,128] shard [2,64,64] => 8 shards / 4 cores
    nd = ttnn.NdShardSpec(ttnn.Shape([2, 64, 64]), grid, ttnn.ShardOrientation.ROW_MAJOR)
    mc = ttnn.MemoryConfig(ttnn.BufferType.L1, nd)
    x = torch.arange(4 * 128 * 128).reshape(4, 128, 128).float().bfloat16()
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    info("nd [4,128,128]/[2,64,64]", t)

    # nd [5,4,160,160] shard [2,3,64,96]
    nd2 = ttnn.NdShardSpec(ttnn.Shape([2, 3, 64, 96]), grid, ttnn.ShardOrientation.ROW_MAJOR)
    mc2 = ttnn.MemoryConfig(ttnn.BufferType.L1, nd2)
    x2 = torch.arange(5 * 4 * 160 * 160).reshape(5, 4, 160, 160).float().bfloat16()
    t2 = ttnn.from_torch(x2, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc2)
    info("nd [5,4,160,160]/[2,3,64,96]", t2)

    # interleaved DRAM input
    xd = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    info("interleaved DRAM [4,128,128]", xd)
finally:
    ttnn.close_device(device)
