import torch, ttnn

d = ttnn.open_device(device_id=0)
print("BANKS", d.dram_grid_size(), d.num_dram_channels() if hasattr(d, "num_dram_channels") else None)
print("GRID", d.compute_with_storage_grid_size())
w = ttnn.from_torch(torch.randn(1536, 4096).bfloat16(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=d)
print("PAGE", w.buffer_page_size(), w.buffer_aligned_page_size() if hasattr(w, "buffer_aligned_page_size") else None)
ttnn.close_device(d)
