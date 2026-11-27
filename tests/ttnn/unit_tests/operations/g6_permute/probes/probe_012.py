import torch, ttnn

device = ttnn.open_device(device_id=0)
g = device.compute_with_storage_grid_size()
print("grid", g.x, g.y)
print("l1 banks", ttnn.get_memory_view(device, ttnn.BufferType.L1).num_banks)
print("dram banks", ttnn.get_memory_view(device, ttnn.BufferType.DRAM).num_banks)
print("l1 align", ttnn.get_l1_alignment(), "dram align", ttnn.get_dram_alignment())
x = ttnn.from_torch(
    torch.randn(1, 128, 256, 32).to(torch.bfloat16),
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG,
)
print("page", x.buffer_page_size(), x.buffer_aligned_page_size(), x.buffer_num_pages())
ttnn.close_device(device)
