import ttnn, torch, math

device = ttnn.open_device(device_id=0)
try:
    workers = ttnn.get_optimal_dram_bank_to_logical_worker_assignment(device, ttnn.NOC.NOC_0)
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in workers])
    print("crs", crs)
    print("cores rm", [(c.x, c.y) for c in ttnn.corerange_to_cores(crs, None, True)])
    # width-sharded output: 8 shards of 2 tiles; write distinct values and check which core holds which shard
    n = 8 * 64
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs, [32, 64], ttnn.ShardOrientation.ROW_MAJOR),
    )
    t = ttnn.from_torch(
        torch.arange(32 * n).reshape(32, n).float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )
    print(t.memory_config())
    # B page order check: compare to_torch roundtrip
    k, nn = 64, 704
    banks = 8
    padded_n = math.ceil(nn / (32 * banks)) * 32 * banks
    g = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
    wmc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(g, (k, padded_n // banks), ttnn.ShardOrientation.ROW_MAJOR),
    )
    w = ttnn.from_torch(
        torch.randn(k, nn), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=wmc
    )
    print("w addr", w.buffer_address(), "pages", w.buffer_num_pages())
finally:
    ttnn.close_device(device)
