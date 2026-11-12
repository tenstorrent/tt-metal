import torch, ttnn

device = ttnn.open_device(device_id=0)
x = ttnn.from_torch(
    torch.randn(1, 1, 64, 1024),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
v = ttnn.device.get_memory_view(device, ttnn.BufferType.L1)
print("BASE", ttnn.get_allocator_base_address(device, ttnn.BufferType.L1))
print("TOT", v.total_bytes_per_bank, v.largest_contiguous_bytes_free_per_bank)
for b in v.block_table[:10]:
    print("BT", b)
ttnn.close_device(device)
