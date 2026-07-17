import ttnn

device = ttnn.open_device(device_id=0)
g = device.compute_with_storage_grid_size()
print("COMPUTE_GRID:", g.x, "x", g.y, "=", g.x * g.y)
ttnn.close_device(device)
