import ttnn

device = ttnn.open_device(device_id=0)
try:
    print("DRAM grid", device.dram_grid_size(), "GRID", device.compute_with_storage_grid_size())
    for name in ("num_dram_channels",):
        if hasattr(device, name):
            print("DRAM", name, getattr(device, name)())
finally:
    ttnn.close_device(device)
