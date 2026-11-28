import ttnn

d = ttnn.open_device(device_id=0)
print(
    "BANKS L1",
    ttnn.get_memory_view(d, ttnn.BufferType.L1).num_banks,
    "DRAM",
    ttnn.get_memory_view(d, ttnn.BufferType.DRAM).num_banks,
    d.compute_with_storage_grid_size(),
)
ttnn.close_device(d)
