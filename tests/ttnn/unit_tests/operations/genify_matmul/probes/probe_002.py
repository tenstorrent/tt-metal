import ttnn

dev = ttnn.open_device(device_id=0)
mv = ttnn.get_memory_view(dev, ttnn.BufferType.L1)
print(
    "MV",
    mv.num_banks,
    mv.total_bytes_per_bank,
    mv.total_bytes_allocated_per_bank,
    mv.largest_contiguous_bytes_free_per_bank,
    ttnn.get_max_worker_l1_unreserved_size(),
    ttnn.get_allocator_base_address(dev, ttnn.BufferType.L1),
)
ttnn.close_device(dev)
