import ttnn

d = ttnn.open_device(device_id=0)
try:
    print("BANKS", d.dram_grid_size(), d.num_dram_channels() if hasattr(d, "num_dram_channels") else "?")
finally:
    ttnn.close_device(d)
