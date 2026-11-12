import ttnn

d = ttnn.open_device(device_id=0)
mv = ttnn.device.get_memory_view(d, ttnn.BufferType.L1)
print(
    "PROBE mv",
    mv.num_banks,
    mv.total_bytes_per_bank,
    mv.total_bytes_free_per_bank,
    mv.largest_contiguous_bytes_free_per_bank,
)
ttnn.close_device(d)
