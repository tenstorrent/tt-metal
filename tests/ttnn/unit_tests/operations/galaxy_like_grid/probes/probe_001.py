import ttnn
d = ttnn.open_device(device_id=0)
g = d.compute_with_storage_grid_size()
print("GRID compute_with_storage", g.x, "x", g.y, "=", g.x * g.y, "cores")
ttnn.close_device(d)
