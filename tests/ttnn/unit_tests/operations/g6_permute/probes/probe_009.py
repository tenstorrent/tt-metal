import ttnn

device = ttnn.open_device(device_id=0)
try:
    g = device.compute_with_storage_grid_size()
    print("GRID", g.x, g.y)
    print(
        "BANKS dram",
        ttnn.get_memory_view(device, ttnn.BufferType.DRAM).num_banks,
        "l1",
        ttnn.get_memory_view(device, ttnn.BufferType.L1).num_banks,
    )
finally:
    ttnn.close_device(device)
