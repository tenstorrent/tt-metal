import torch, ttnn, math

torch.manual_seed(42)
Q = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
K = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
V = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)

mask = torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16)
mask.masked_fill_(torch.triu(torch.ones(32, 32, dtype=torch.bool), diagonal=1), float("-inf"))

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

# Check mask values
mask_back = ttnn.to_torch(ttnn_mask)
print(f"Mask[0,0,:4,:4]:\n{mask_back[0,0,:4,:4]}")
print(f"Mask has -inf: {(mask_back == float('-inf')).any()}")
print(f"Mask addr: {ttnn_mask.buffer_address()}")
print(f"Q addr: {ttnn_Q.buffer_address()}")

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

output = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V, attn_mask=ttnn_mask, scale=0.1)
result = ttnn.to_torch(output)

print(f"Output[0,0,:4,:4]:\n{result[0,0,:4,:4]}")
print(f"Output has Inf: {torch.isinf(result.float()).any()}")
print(f"Output has NaN: {torch.isnan(result.float()).any()}")

# Check which rows have Inf
inf_rows = torch.isinf(result.float()).any(dim=-1)
print(f"Inf rows: {inf_rows.nonzero().flatten().tolist()}")

ttnn.close_device(device)
