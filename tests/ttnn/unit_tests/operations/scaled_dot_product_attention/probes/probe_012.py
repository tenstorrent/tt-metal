import ttnn

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"GRID: cols(x)={grid.x} rows(y)={grid.y}  total={grid.x*grid.y}")
ttnn.close_device(device)
