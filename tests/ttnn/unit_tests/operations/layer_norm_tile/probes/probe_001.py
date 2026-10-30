import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    g = ttnn.from_torch(
        torch.randn(768).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    print("tile gamma", g.shape, g.padded_shape, g.buffer_num_pages(), g.buffer_page_size())
    r = ttnn.from_torch(
        torch.randn(768).reshape(1, 1, 24, 32).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    print("rm gamma", r.shape, r.buffer_num_pages(), r.buffer_page_size())
    x = ttnn.from_torch(
        torch.randn(124, 1, 33, 768).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    mc = x.memory_config()
    print("x", x.shape, x.padded_shape, x.buffer_num_pages(), mc.memory_layout, mc.buffer_type)
    print("grid", device.compute_with_storage_grid_size(), "arch", device.arch())
finally:
    ttnn.close_device(device)
