import ttnn

device = ttnn.open_device(device_id=0)
try:
    g = device.compute_with_storage_grid_size()
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
    base = ttnn.TensorSpec.with_padded_shape(
        ttnn.Shape([50, 256]),
        ttnn.Shape([64, 256]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.L1_MEMORY_CONFIG,
        ttnn.Tile([32, 32]),
    )
    for name in ("width_sharded", "height_sharded", "block_sharded"):
        try:
            sp = getattr(base, name)(grid, ttnn.ShardOrientation.ROW_MAJOR)
            t = ttnn.allocate_tensor_on_device(sp, device)
            print(name, "->", t.memory_config())
            print("   logical", list(t.shape), "padded", list(t.padded_shape))
        except Exception as e:
            print(name, "FAILED", type(e).__name__, str(e)[:160])
    # and the 8,1,49,2048 case
    base2 = ttnn.TensorSpec.with_padded_shape(
        ttnn.Shape([8, 1, 49, 2048]),
        ttnn.Shape([8, 1, 64, 2048]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        ttnn.L1_MEMORY_CONFIG,
        ttnn.Tile([32, 32]),
    )
    sp2 = base2.width_sharded(grid, ttnn.ShardOrientation.ROW_MAJOR)
    t2 = ttnn.allocate_tensor_on_device(sp2, device)
    print("big width_sharded ->", t2.memory_config())
    print("   padded", list(t2.padded_shape))
finally:
    ttnn.close_device(device)
