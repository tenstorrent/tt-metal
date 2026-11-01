import ttnn

d = ttnn.open_device(device_id=0)
print("BANKS", d.num_dram_channels() if hasattr(d, "num_dram_channels") else None, d.dram_grid_size())
ttnn.close_device(d)
