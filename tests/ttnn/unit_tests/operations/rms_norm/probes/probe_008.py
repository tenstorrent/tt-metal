import torch, ttnn, sys

sys.path.insert(0, ".")
from eval.sharding import auto_shard_config

device = ttnn.open_device(device_id=0)


def rng(r):
    for a in ("start", "start_coord"):
        if hasattr(r, a):
            s = getattr(r, a)
            e = getattr(r, "end" if a == "start" else "end_coord")
            return (s.x, s.y, e.x, e.y)
    return dir(r)


try:
    for shape, ml, layout, dt in [
        ((1, 1, 256, 512), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ((1, 1, 32, 2048), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ((1, 1, 256, 512), ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ((1, 1, 224, 1000), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ((1, 1, 224, 1000), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
        ((1, 1, 100, 736), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
        ((1, 1, 3232, 96), ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TILE_LAYOUT, ttnn.bfloat16),
        ((1, 1, 32, 4096), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TILE_LAYOUT, ttnn.bfloat16),
    ]:
        mc = auto_shard_config(list(shape), ml, layout=layout, dtype=dt, device=device)
        ss = mc.shard_spec
        cores = ttnn.corerange_to_cores(ss.grid, None, ss.orientation == ttnn.ShardOrientation.ROW_MAJOR)
        t = ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16), dtype=dt, layout=layout, device=device, memory_config=mc
        )
        cbd = ttnn.cb_descriptor_from_sharded_tensor(1, t)
        print(
            shape,
            str(ml).split(".")[-1],
            str(layout).split(".")[-1],
            "shard",
            list(ss.shape),
            "ncores",
            len(cores),
            "bbox",
            rng(ss.grid.bounding_box()),
            "addr",
            hex(t.buffer_address()),
            "cbsize",
            cbd.total_size,
            "page",
            cbd.format_descriptors[0].page_size,
        )
        print(
            "   cores",
            [(c.x, c.y) for c in cores[:4]],
            "..",
            (cores[-1].x, cores[-1].y),
            "ranges",
            [rng(r) for r in ss.grid.ranges()],
        )
        t.deallocate()
finally:
    ttnn.close_device(device)
