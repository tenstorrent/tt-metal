import ttnn, torch

device = ttnn.open_device(device_id=0)
try:
    print("DRAM_ALIGN", ttnn.get_dram_alignment(), "L1_ALIGN", ttnn.get_l1_alignment())
    x = ttnn.from_torch(
        torch.zeros(16, 12, 128, 128),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    print("PAGES", x.buffer_num_pages(), "PAGE", x.buffer_page_size(), "ALIGNED", x.buffer_aligned_page_size())
    y = ttnn.from_torch(
        torch.zeros(1, 1),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    print("PAGES11", y.buffer_num_pages(), y.buffer_page_size(), y.buffer_aligned_page_size(), y.padded_shape)
    z = ttnn.from_torch(
        torch.zeros(5, 1, 32, 256),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    print("BFP8", z.buffer_num_pages(), z.buffer_page_size(), z.buffer_aligned_page_size())
    r = ttnn.from_torch(
        torch.zeros(1, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    print("RM1", r.buffer_num_pages(), r.buffer_page_size(), r.buffer_aligned_page_size())
    w = ttnn.from_torch(
        torch.zeros(1, 1, 1, 256, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    print("BF16_1x256", w.buffer_num_pages(), w.padded_shape)
    print("SPLIT", ttnn.split_work_to_cores.__doc__[:900] if ttnn.split_work_to_cores.__doc__ else None)
finally:
    ttnn.close_device(device)
