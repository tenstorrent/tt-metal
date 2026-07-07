import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.CreateDevice(0)

B, H, S, D = 1, 1, 1024, 64
q = torch.ones(B, H, S, D, dtype=torch.bfloat16)
k = torch.ones(B, H, S, D, dtype=torch.bfloat16)
v = torch.ones(B, H, S, D, dtype=torch.bfloat16)

q_t = ttnn.from_torch(
    q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
k_t = ttnn.from_torch(
    k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
v_t = ttnn.from_torch(
    v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)

output = scaled_dot_product_attention(q_t, k_t, v_t)
result = ttnn.to_torch(output).float()

# Check which Q blocks (rows of 32) are zero
for qb in range(32):
    row_sum = result[0, 0, qb * 32 : (qb + 1) * 32, :].sum().item()
    if row_sum < 32 * 64:  # not all ones
        print(f"Q block {qb}: sum={row_sum} (expected {32*64}), first row: {result[0,0,qb*32,:5]}")
    else:
        print(f"Q block {qb}: OK (sum={row_sum})")

ttnn.CloseDevice(device)
