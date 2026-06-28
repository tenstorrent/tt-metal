import torch, ttnn, math

torch.manual_seed(42)
Q = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
K = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
V = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)

# Use a simple mask with -100 instead of -inf to avoid -inf issues
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

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

output = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V, attn_mask=ttnn_mask, scale=0.1)
result = ttnn.to_torch(output)

# Reference
scale = 0.1
scores = Q @ K.transpose(-2, -1) * scale + mask
exp_scores = torch.exp(scores - scores.max(dim=-1, keepdim=True).values)
weights = exp_scores / exp_scores.sum(dim=-1, keepdim=True)
expected = weights @ V

print(f"Output[0,0,:4,:4]:\n{result[0,0,:4,:4]}")
print(f"Expected[0,0,:4,:4]:\n{expected[0,0,:4,:4]}")
print(f"Max diff: {(result.float() - expected.float()).abs().max()}")

# Compute PCC
output_flat = result.float().flatten()
expected_flat = expected.float().flatten()
output_centered = output_flat - output_flat.mean()
expected_centered = expected_flat - expected_flat.mean()
numerator = (output_centered * expected_centered).sum()
denominator = torch.sqrt((output_centered**2).sum()) * torch.sqrt((expected_centered**2).sum())
pcc = (numerator / denominator).item() if denominator > 0 else 0.0
print(f"PCC: {pcc:.6f}")

ttnn.close_device(device)
