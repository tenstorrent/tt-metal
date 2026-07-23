import ttnn

g = device.compute_with_storage_grid_size()
print("COMPUTE_GRID", g.x, g.y, "num_cores", g.x * g.y)
