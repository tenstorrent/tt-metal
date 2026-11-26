import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    g = device.compute_with_storage_grid_size()
    print("grid", g.x, g.y)
    info = ttnn._ttnn.reports.get_device_info(device)
    print("l1_num_banks", info.l1_num_banks, "dram banks", getattr(info, "num_dram_banks", None))
    print("device id", device.id(), "dram grid", device.dram_grid_size())
    print("l1 align", ttnn.get_l1_alignment() if hasattr(ttnn, "get_l1_alignment") else None)
    # shard spec fields
    grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(0, 2)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 1), ttnn.CoreCoord(5, 1)),
            ttnn.CoreRange(ttnn.CoreCoord(1, 2), ttnn.CoreCoord(1, 2)),
        ]
    )
    print("ranges order", [(r.start.x, r.start.y) for r in grid.ranges()])
    merged = ttnn.CoreRangeSet([]).merge(grid) if True else None
    print("merged", merged)
    x = torch.randn(1, 16, 1, 256).to(torch.bfloat16)
    t = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    print(
        "addr",
        t.buffer_address(),
        "pg",
        t.buffer_page_size(),
        t.buffer_aligned_page_size(),
        t.buffer_num_pages(),
        "padded",
        t.padded_shape,
    )
    mc = t.memory_config()
    print("mc", mc.buffer_type, mc.memory_layout, mc.shard_spec, mc.is_sharded())
    a = ttnn.TensorAccessorArgs(t).get_compile_time_args()
    print("dram interleaved ta args", list(a))
    sp = ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))]),
        (32, 544),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    print("shard fields", sp.grid, sp.shape, sp.orientation, sp.num_cores())
    smc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, sp)
    t2 = ttnn.from_torch(
        torch.randn(1, 1, 16, 4352).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=smc,
    )
    print("sharded ta args", list(ttnn.TensorAccessorArgs(t2).get_compile_time_args()))
    o = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 16, 4352]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, smc)
    print("alloc ok", o.memory_config())
    li = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    print("l1 interleaved ta args", list(ttnn.TensorAccessorArgs(li).get_compile_time_args()), li.buffer_address())
    s = ttnn.split_work_to_cores(g, 1000, row_wise=True)
    print("split", s[0], s[1], s[2], s[3], s[4], s[5])
    print(ttnn.corerange_to_cores(s[2], row_wise=True)[:3])
finally:
    ttnn.close_device(device)
