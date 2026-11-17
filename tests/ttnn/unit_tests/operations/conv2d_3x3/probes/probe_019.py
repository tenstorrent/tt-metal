import torch, ttnn
from eval.sharding import auto_shard_config

device = ttnn.open_device(device_id=0)
try:
    g = device.compute_with_storage_grid_size()
    print("grid", g.x, g.y)
    for (H, W, C), layout, dt in [
        ((16, 16, 64), ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
        ((16, 16, 64), ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ((8, 16, 2048), ttnn.TILE_LAYOUT, ttnn.bfloat8_b),
        ((8, 16, 2048), ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
        ((16, 34, 32), ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
        ((14, 16, 48), ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ((16, 16, 8), ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ((32, 64, 128), ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
    ]:
        mc = auto_shard_config(
            [1, 1, H * W, C], ttnn.TensorMemoryLayout.BLOCK_SHARDED, layout=layout, dtype=dt, device=device
        )
        x = torch.arange(H * W * C, dtype=torch.float32).reshape(1, 1, H * W, C) % 251
        t = ttnn.from_torch(x.to(torch.bfloat16), dtype=dt, layout=layout, device=device, memory_config=mc)
        s = mc.shard_spec
        print(
            (H, W, C),
            layout,
            "shard",
            s.shape,
            "grid",
            s.grid.bounding_box(),
            "page",
            t.buffer_aligned_page_size(),
            "padded",
            t.padded_shape,
        )
    # Output BLOCK with fewer width shards than grid cols
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 1))})  # 4 cols x 2 rows
    omc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [64, 32], ttnn.ShardOrientation.ROW_MAJOR),
    )
    y = torch.randn(1, 1, 128, 64)
    yt = ttnn.from_torch(
        y.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=omc
    )
    back = ttnn.to_torch(yt).float()
    print("fewer-width-shards roundtrip max diff", (back - y.to(torch.bfloat16).float()).abs().max().item())
    a = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 128, 64]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, omc)
    print("alloc ok", a.memory_config())
finally:
    ttnn.close_device(device)
