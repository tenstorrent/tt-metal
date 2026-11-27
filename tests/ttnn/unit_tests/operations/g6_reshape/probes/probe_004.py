import torch, ttnn

d = ttnn.open_device(device_id=0)
print(
    "grid",
    d.compute_with_storage_grid_size(),
    "dram_align",
    ttnn.get_dram_alignment() if hasattr(ttnn, "get_dram_alignment") else None,
)
x = ttnn.from_torch(torch.randn(1, 1, 32, 32).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=d)
y = ttnn.add(x, x)
print(ttnn.to_torch(y).shape)
t = ttnn.from_torch(torch.arange(185).reshape(1, 185).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=d)
print(
    "shape",
    t.shape,
    "padded",
    t.padded_shape,
    "pages",
    t.buffer_num_pages() if hasattr(t, "buffer_num_pages") else "na",
    "elsize",
    t.element_size(),
)
t1 = ttnn.from_torch(torch.arange(16).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=d)
print("rank1", t1.shape, t1.padded_shape)
ttnn.close_device(d)
