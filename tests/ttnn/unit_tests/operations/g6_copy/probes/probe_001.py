import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    g = device.compute_with_storage_grid_size()
    print("GRID", g.x, g.y)
    for name in ("dram_grid_size",):
        try:
            d = getattr(device, name)()
            print("DRAM_GRID", d.x, d.y)
        except Exception as e:
            print(name, "ERR", e)
    try:
        print("NUM_DRAM_CH", device.num_dram_channels())
    except Exception as e:
        print("num_dram_channels ERR", e)
    print("ALIGN dram", ttnn.get_dram_alignment(), "l1", ttnn.get_l1_alignment())
    cases = [
        ((16, 12, 128, 128), ttnn.float32, ttnn.TILE_LAYOUT, torch.float32),
        ((1, 1), ttnn.float32, ttnn.TILE_LAYOUT, torch.float32),
        ((1, 1, 1, 256), ttnn.float32, ttnn.TILE_LAYOUT, torch.float32),
        ((5, 1, 32, 256), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, torch.float32),
        ((16, 3, 2560), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, torch.bfloat16),
        ((1,), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32),
        ((7, 1, 1, 32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32),
    ]
    for shape, dt, lay, tdt in cases:
        t = ttnn.from_torch(
            torch.zeros(shape, dtype=tdt), dtype=dt, layout=lay, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        a = ttnn.TensorAccessorArgs(t)
        print(
            "T",
            shape,
            dt,
            lay,
            "pages",
            t.buffer_num_pages(),
            "page",
            t.buffer_page_size(),
            "aligned",
            t.buffer_aligned_page_size(),
            "ct",
            len(a.get_compile_time_args()),
            "crt",
            len(a.get_common_runtime_args()),
        )
    r = ttnn.split_work_to_cores(g, 3072, row_wise=True)
    print("SPLIT3072", r[0], r[4], r[5], r[2].num_cores() if hasattr(r[2], "num_cores") else r[2], r[3])
    r = ttnn.split_work_to_cores(g, 192, row_wise=True)
    print("SPLIT192", r[0], r[4], r[5], r[1])
finally:
    ttnn.close_device(device)
