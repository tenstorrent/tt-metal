import ttnn

device = ttnn.open_device(device_id=0)
g = device.compute_with_storage_grid_size()
print("grid", g.x, g.y)
print("x map:", [(x, device.worker_core_from_logical_core(ttnn.CoreCoord(x, 0)).x) for x in range(g.x)])
print("y map:", [(y, device.worker_core_from_logical_core(ttnn.CoreCoord(0, y)).y) for y in range(g.y)])
ttnn.close_device(device)
