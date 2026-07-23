import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    g = device.compute_with_storage_grid_size()
    print("GRID:", g.x, "x", g.y, "=", g.x * g.y, "cores")
finally:
    ttnn.close_device(device)
