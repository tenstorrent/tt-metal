import ttnn

d = ttnn.open_device(device_id=0)
g = d.compute_with_storage_grid_size()
print(f"EB_GRID {g.x}x{g.y}", flush=True)
ttnn.close_device(d)
