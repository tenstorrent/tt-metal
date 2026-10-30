import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    g = device.compute_with_storage_grid_size()
    print("grid", g.x, g.y)
    print("l1_alignment", ttnn.get_l1_alignment(), "dram_alignment", ttnn.get_dram_alignment())
    t1 = ttnn.from_torch(
        torch.randn(768, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    print("gamma tile", t1.shape, t1.padded_shape, t1.buffer_num_pages(), t1.buffer_page_size())
    t2 = ttnn.from_torch(
        torch.randn(1, 1, 24, 32, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    print("gamma rm", t2.shape, t2.padded_shape, t2.buffer_num_pages(), t2.buffer_page_size())
    x = ttnn.from_torch(
        torch.randn(2, 1, 47, 768, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    print("x", x.shape, x.padded_shape, x.buffer_num_pages())
    xt = ttnn.to_torch(x)
    print("roundtrip", xt.shape)
    print("cfg", ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi).fp32_dest_acc_en)
finally:
    ttnn.close_device(device)
