import ttnn, torch

device = ttnn.open_device(device_id=0)
t = ttnn.from_torch(
    torch.zeros(1, 1, 1024, 1024),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
mv = ttnn.get_memory_view(device, ttnn.BufferType.L1)
print(mv.block_table, mv.total_bytes_per_bank, mv.largest_contiguous_bytes_free_per_bank)
print("tensor addr", t.buffer_address(), ttnn.device.get_allocator_base_address(device, ttnn.BufferType.L1))
ttnn.close_device(device)
