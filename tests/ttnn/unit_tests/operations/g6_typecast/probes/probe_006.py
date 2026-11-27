import torch, ttnn

device = ttnn.open_device(device_id=0)
for shape, dt, tdt in [
    ((1, 2048, 2048, 3), ttnn.bfloat16, torch.bfloat16),
    ((1, 3844), ttnn.uint32, torch.int64),
    ((1,), ttnn.int32, torch.int32),
]:
    x = torch.zeros(shape, dtype=tdt)
    t = ttnn.from_torch(x, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    print(shape, t.buffer_page_size(), t.buffer_aligned_page_size(), t.buffer_num_pages(), t.padded_shape)
o = ttnn.allocate_tensor_on_device(
    ttnn.Shape([1, 2048, 2048, 3]), ttnn.uint8, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
)
print(o.buffer_page_size(), o.buffer_aligned_page_size(), o.buffer_num_pages())
print(device.compute_with_storage_grid_size())
try:
    print(device.num_banks(ttnn.BufferType.DRAM), device.num_banks(ttnn.BufferType.L1))
except Exception as e:
    print("ERR", e)
ttnn.close_device(device)
