import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    print("arch", device.arch())
    g = device.compute_with_storage_grid_size()
    print("grid", g.x, g.y)
    print("dram banks", device.num_dram_channels() if hasattr(device, "num_dram_channels") else "n/a")
    print("dram align", ttnn.get_dram_alignment(), "l1 align", ttnn.get_l1_alignment())
    for shp, dt in [
        ((16,), ttnn.bfloat16),
        ((1, 185), ttnn.float32),
        ((16, 1, 64), ttnn.bfloat16),
        ((4096, 1), ttnn.int32),
    ]:
        lay = ttnn.TILE_LAYOUT if len(shp) != 2 or dt != ttnn.int32 else ttnn.ROW_MAJOR_LAYOUT
        t = ttnn.from_torch(
            torch.zeros(shp).to(torch.float32 if dt != ttnn.int32 else torch.int32),
            dtype=dt,
            layout=lay,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        print(
            shp,
            dt,
            lay,
            "padded",
            list(t.padded_shape),
            "page",
            t.buffer_page_size(),
            "aligned",
            t.buffer_aligned_page_size(),
            "pages",
            t.buffer_num_pages(),
            "addr%64",
            t.buffer_address() % 64,
        )
    print("has split_work_to_cores", hasattr(ttnn, "split_work_to_cores"))
    import inspect

    print(ttnn.split_work_to_cores.__doc__[:600] if ttnn.split_work_to_cores.__doc__ else "")
finally:
    ttnn.close_device(device)
