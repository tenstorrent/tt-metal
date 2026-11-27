import ttnn

device = ttnn.open_device(device_id=0)
try:
    g = device.compute_with_storage_grid_size()
    print("PROBE grid", g.x, g.y)
    print("PROBE dram_align", ttnn.get_dram_alignment(), "l1_align", ttnn.get_l1_alignment())
    print("PROBE dram banks", device.dram_grid_size(), getattr(device, "num_dram_channels", lambda: None)())
    print("PROBE arch", device.arch())
finally:
    ttnn.close_device(device)
