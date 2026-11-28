import ttnn

dev = ttnn.open_device(device_id=0)
try:
    print("arch", dev.arch())
    print("grid", dev.compute_with_storage_grid_size())
    print("dram_grid", dev.dram_grid_size())
    print("num_dram_channels", dev.num_dram_channels() if hasattr(dev, "num_dram_channels") else "n/a")
    print("dram_align", ttnn.get_dram_alignment(), "l1_align", ttnn.get_l1_alignment())
    t = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 64]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
    )
    print("bfp8 page", t.buffer_page_size(), t.buffer_aligned_page_size(), t.buffer_num_pages())
    print("l1 size", dev.l1_size_per_core() if hasattr(dev, "l1_size_per_core") else "n/a")
    for i in range(8):
        try:
            print(
                "dram view",
                i,
                dev.logical_core_from_dram_channel(i) if hasattr(dev, "logical_core_from_dram_channel") else "",
            )
        except Exception as e:
            print("err", e)
            break
    print([m for m in dir(dev) if "dram" in m.lower() or "bank" in m.lower()])
finally:
    ttnn.close_device(dev)
