import ttnn

device = ttnn.open_device(device_id=0)
try:
    g = device.compute_with_storage_grid_size()
    print("PROBE grid", g.x, g.y)
    print("PROBE dram_align", ttnn.get_dram_alignment(), "l1_align", ttnn.get_l1_alignment())
    t = ttnn.from_torch(
        __import__("torch").zeros(1, 1, 1, 512, dtype=__import__("torch").bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    print(
        "PROBE page", t.buffer_page_size(), t.buffer_num_pages(), t.padded_shape, t.tile.tile_shape, t.tile.face_shape
    )
finally:
    ttnn.close_device(device)
