import torch, ttnn, sys

sys.path.insert(0, ".")
from eval.sharding import auto_shard_config, shard_config

device = ttnn.open_device(device_id=0)
try:
    g = device.compute_with_storage_grid_size()
    print("GRID", g.x, g.y, "arch", ttnn.get_arch_name())
    print("l1_align", ttnn._ttnn.device.get_l1_alignment())
    for shape, ml, layout, dt in [
        ((1, 1, 256, 512), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ((1, 1, 32, 2048), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ((1, 1, 256, 512), ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ((1, 1, 224, 1000), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ((1, 1, 224, 1000), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
        ((1, 1, 100, 736), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
    ]:
        mc = auto_shard_config(list(shape), ml, layout=layout, dtype=dt, device=device)
        ss = mc.shard_spec
        cores = ttnn.corerange_to_cores(ss.grid, None, ss.orientation == ttnn.ShardOrientation.ROW_MAJOR)
        t = ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16), dtype=dt, layout=layout, device=device, memory_config=mc
        )
        print(
            shape,
            str(ml).split(".")[-1],
            str(layout).split(".")[-1],
            "shard",
            list(ss.shape),
            "ncores",
            len(cores),
            "bbox",
            ss.grid.bounding_box(),
            "addr",
            hex(t.buffer_address()),
        )
        print("   cores[:6]", [(c.x, c.y) for c in cores[:6]], "... last", (cores[-1].x, cores[-1].y))
        print("   ranges", [(r.start_coord.x, r.start_coord.y, r.end_coord.x, r.end_coord.y) for r in ss.grid.ranges()])
        cbd = ttnn.cb_descriptor_from_sharded_tensor(1, t)
        print(
            "   cb total_size",
            cbd.total_size,
            "page",
            cbd.format_descriptors[0].page_size,
            "has_buffer",
            cbd.has_buffer(),
        )
        t.deallocate()
finally:
    ttnn.close_device(device)
