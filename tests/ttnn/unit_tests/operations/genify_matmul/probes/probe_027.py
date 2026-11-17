import ttnn

device = ttnn.open_device(device_id=0)
v = ttnn.get_memory_view(device, ttnn.BufferType.L1)
print("L1 banks", v.num_banks, v.total_bytes_per_bank)
v = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
print("DRAM banks", v.num_banks, v.total_bytes_per_bank)
ttnn.close_device(device)
