import ttnn

device = ttnn.open_device(device_id=0)
g = device.compute_with_storage_grid_size()
for y in range(g.y):
    for x in range(g.x):
        c = device.worker_core_from_logical_core(ttnn.CoreCoord(x, y))
        print("MAP", x, y, c.x, c.y)
ttnn.close_device(device)
