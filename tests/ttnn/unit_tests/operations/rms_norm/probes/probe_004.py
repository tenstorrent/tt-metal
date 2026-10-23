import torch, ttnn
from eval.sharding import auto_shard_config

device = ttnn.open_device(device_id=0)
g = device.compute_with_storage_grid_size()
print("grid", g.x, g.y, "l1_align", ttnn._ttnn.device.get_l1_alignment(), "dram_align", ttnn.get_dram_alignment())
for shape, ml, layout, dt in [
    ((1, 1, 32, 50), ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
    ((1, 1, 416, 1184), ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
    ((1, 1, 416, 1184), ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.ROW_MAJOR_LAYOUT, ttnn.bfloat16),
    ((1, 1, 40, 40), ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TILE_LAYOUT, ttnn.bfloat16),
]:
    mc = auto_shard_config(list(shape), ml, layout=layout, dtype=dt, device=device)
    t = ttnn.from_torch(torch.randn(shape), dtype=dt, layout=layout, device=device, memory_config=mc)
    ss = t.memory_config().shard_spec
    print(
        shape,
        ml,
        layout,
        "shard",
        ss.shape,
        ss.grid,
        ss.orientation,
        "addr",
        t.buffer_address(),
        "padded",
        t.padded_shape,
    )
    b = t.buffer() if hasattr(t, "buffer") else None
    print("  buffer attrs:", [a for a in dir(b) if not a.startswith("_")][:40] if b is not None else None)
    try:
        print("  page", b.page_size(), "aligned", b.aligned_page_size(), "per_bank", b.aligned_size_per_bank())
    except Exception as e:
        print("  err", e)
    cb = ttnn.cb_descriptor_from_sharded_tensor(0, t)
    print("  cb total", cb.total_size, [(f.page_size) for f in cb.format_descriptors])
print(
    ttnn.corerange_to_cores(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(2, 1))}), row_wise=True
    )
)
ttnn.close_device(device)
