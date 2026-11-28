import ttnn

d = ttnn.open_device(device_id=0)
print("DRAMGRID", d.dram_grid_size())
ttnn.close_device(d)
