import ttnn

d = ttnn.open_device(device_id=0)
print("PROBE unreserved", ttnn.device.get_max_worker_l1_unreserved_size())
print("PROBE base", ttnn.device.get_allocator_base_address(d, ttnn.BufferType.L1))
mv = ttnn.device.get_memory_view(d, ttnn.BufferType.L1)
print("PROBE mv", [a for a in dir(mv) if not a.startswith("_")])
print("PROBE mv", mv.num_banks, mv.total_bytes_per_bank, mv.total_bytes_allocatable_per_bank)
ttnn.close_device(d)
