import ttnn

g = device.compute_with_storage_grid_size()
print("GRID:", g.x, "x", g.y, "=", g.x * g.y, "cores")
