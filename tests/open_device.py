import ttnn

device_id = 0

dev1 = ttnn.open_device(device_id=device_id)
grid1 = dev1.compute_with_storage_grid_size()
ttnn.close_device(dev1)

dev2 = ttnn.CreateDevice(device_id=device_id)
grid2 = dev2.compute_with_storage_grid_size()
ttnn.close_device(dev2)

print(f"open_device   compute_grid: {grid1}")
print(f"CreateDevice  compute_grid: {grid2}")
print(f"grids_equal: {grid1 == grid2}")
