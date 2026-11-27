import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    g = device.compute_with_storage_grid_size()
    print("grid", g.x, g.y)
    info = ttnn._ttnn.reports.get_device_info(device)
    print("l1_num_banks", info.l1_num_banks, "num_compute_cores", info.num_compute_cores)
    print(ttnn.split_work_to_cores(g, 250, row_wise=True))
    crs = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 2)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 3), ttnn.CoreCoord(2, 3)),
        ]
    )
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs, (32, 256), ttnn.ShardOrientation.ROW_MAJOR),
    )
    t = ttnn.from_torch(
        torch.randn(1, 1, 16, 8704).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )
    cores = ttnn.get_optimal_worker_cores_for_sharded_tensor(t)
    print(type(cores), len(cores), cores[:3], cores[-2:])
    print("ranges", [(r.start.x, r.start.y, r.end.x, r.end.y) for r in mc.shard_spec.grid.ranges()])
    print("aligned page", t.buffer_aligned_page_size(), "addr", t.buffer_address(), "padded", t.padded_shape)
    print("dram align", ttnn.get_dram_alignment(), "l1 align", ttnn.get_l1_alignment())
    print("ta", len(ttnn.TensorAccessorArgs(t).get_compile_time_args()))
    crs2 = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in [(0, 2), (1, 2), (5, 1), (5, 2)]]
    )
    print("ranges2", [(r.start.x, r.start.y, r.end.x, r.end.y) for r in crs2.ranges()], crs2.bounding_box())
    print(device.id(), hasattr(t, "is_sharded"), t.is_sharded(), t.memory_config().is_sharded())
    print(ttnn.corerange_to_cores if hasattr(ttnn, "corerange_to_cores") else "no c2c")
finally:
    ttnn.close_device(device)
