import ttnn

device = ttnn.open_device(device_id=0)
print("GRID", device.compute_with_storage_grid_size())
print("DRAM", device.dram_grid_size())
ttnn.close_device(device)
