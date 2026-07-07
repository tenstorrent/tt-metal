import ttnn

device = ttnn.open_device(device_id=0)
grid = device.compute_with_storage_grid_size()
print(f"Grid size: x={grid.x}, y={grid.y}")
print(f"Total worker cores: {grid.x * grid.y}")
ttnn.close_device(device)
