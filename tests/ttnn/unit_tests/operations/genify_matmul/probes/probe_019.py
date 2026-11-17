import ttnn, torch, math

device = ttnn.open_device(device_id=0)
try:
    print("grid", device.compute_with_storage_grid_size(), "dram", device.dram_grid_size())
    for noc in (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1):
        cores = ttnn.get_optimal_dram_bank_to_logical_worker_assignment(device, noc)
        print(noc, [(c.x, c.y) for c in cores])
    # B dram-width-sharded page layout check
    k, n = 64, 704
    banks = device.dram_grid_size().x
    padded_n = math.ceil(n / (32 * banks)) * 32 * banks
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(grid, (k, padded_n // banks), ttnn.ShardOrientation.ROW_MAJOR),
    )
    w = ttnn.from_torch(
        torch.randn(k, n), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
    )
    print(
        "w pages",
        w.buffer_num_pages(),
        "aligned page",
        w.buffer_aligned_page_size(),
        "tile",
        ttnn.tile_size(ttnn.bfloat4_b),
    )
    print(w.memory_config())
    a = ttnn.from_torch(
        torch.randn(1, 1, 32, 2048),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.num_cores_to_corerangeset(32, device.compute_with_storage_grid_size(), True),
                (32, 64),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    sg = a.memory_config().shard_spec.grid
    print("a grid", sg, a.buffer_address())
    print("L1_WS", ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG.shard_spec)
    print([x for x in dir(ttnn) if "Mcast" in x])
finally:
    ttnn.close_device(device)
