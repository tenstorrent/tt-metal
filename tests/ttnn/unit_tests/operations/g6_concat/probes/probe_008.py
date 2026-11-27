import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    t = ttnn.from_torch(
        torch.randn(1, 1, 33, 64).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    print("dram_align", ttnn.get_dram_alignment(), "l1_align", ttnn.get_l1_alignment())
    print("ct args", list(ttnn.TensorAccessorArgs(t).get_compile_time_args()))
    print("page", t.buffer_page_size(), t.buffer_num_pages(), t.buffer_aligned_page_size(), t.padded_shape, t.shape)
    g = device.compute_with_storage_grid_size()
    print("grid", g.x, g.y)
    r = ttnn.split_work_to_cores(g, 1000, row_wise=True)
    print(r[0], r[4], r[5], r[2].num_cores(), r[3].num_cores())
    print(ttnn.grid_to_cores(3, g.x, g.y, True))
    print(t.memory_config().buffer_type, t.memory_config().memory_layout, t.memory_config().is_sharded())
    print(ttnn.CBFormatDescriptor, ttnn.ProgramDescriptor().custom_program_hash)
    print(t.buffer_address())
finally:
    ttnn.close_device(device)
