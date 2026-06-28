import torch, ttnn, math

torch.manual_seed(42)
Q = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
K = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
V = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
mask = torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16)
mask.masked_fill_(torch.triu(torch.ones(32, 32, dtype=torch.bool), diagonal=1), -100.0)

device = ttnn.open_device(device_id=0)
ttnn_Q = ttnn.from_torch(
    Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
ttnn_K = ttnn.from_torch(
    K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
ttnn_V = ttnn.from_torch(
    V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
ttnn_mask = ttnn.from_torch(
    mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)

print(f"Q addr: {ttnn_Q.buffer_address()}")
print(f"K addr: {ttnn_K.buffer_address()}")
print(f"V addr: {ttnn_V.buffer_address()}")
print(f"Mask addr: {ttnn_mask.buffer_address()}")
print(f"Q num_pages: {ttnn_Q.buffer_num_pages()}")
print(f"Mask num_pages: {ttnn_mask.buffer_num_pages()}")

# Check if mask is at same addr as Q/K/V
print(f"Mask == Q addr? {ttnn_mask.buffer_address() == ttnn_Q.buffer_address()}")
print(f"Mask == K addr? {ttnn_mask.buffer_address() == ttnn_K.buffer_address()}")

ttnn.close_device(device)
