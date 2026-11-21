import torch, ttnn
dev = ttnn.open_device(device_id=0)
try:
    print("grid", dev.compute_with_storage_grid_size())
    print("l1 unreserved", ttnn.get_max_worker_l1_unreserved_size() if hasattr(ttnn, "get_max_worker_l1_unreserved_size") else "n/a")
    print("l1 align", ttnn.get_l1_alignment(), "dram align", ttnn.get_dram_alignment())
    for shape, layout, mc in [((4,128,47), ttnn.ROW_MAJOR_LAYOUT, ttnn.L1_MEMORY_CONFIG), ((4,128,47), ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG), ((512,), ttnn.TILE_LAYOUT, ttnn.L1_MEMORY_CONFIG), ((1,64,400), ttnn.TILE_LAYOUT, ttnn.L1_MEMORY_CONFIG)]:
        t = ttnn.from_torch(torch.randn(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=layout, device=dev, memory_config=mc)
        print(shape, layout, "pages", t.buffer_num_pages(), "psize", t.buffer_page_size(), "aligned", t.buffer_aligned_page_size() if hasattr(t,"buffer_aligned_page_size") else "?", "padded", t.padded_shape, "tile", t.tile.tile_shape, t.memory_config().buffer_type)
    spec = ttnn.TensorSpec((1,4,16,400), ttnn.bfloat16, ttnn.TILE_LAYOUT, buffer_type=ttnn.BufferType.L1) if False else None
    import inspect
    print(ttnn.TensorSpec.__init__.__doc__)
    print(ttnn.allocate_tensor_on_device.__doc__)
finally:
    ttnn.close_device(dev)
