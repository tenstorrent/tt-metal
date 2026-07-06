import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

torch.manual_seed(42)
q = torch.ones(1, 1, 32, 32, dtype=torch.bfloat16)
k = torch.ones(1, 1, 32, 32, dtype=torch.bfloat16)
v = torch.ones(1, 1, 32, 32, dtype=torch.bfloat16)

ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
print(f"Ref first 4: {ref[0,0,0,:4]}")

device = ttnn.open_device(device_id=0)
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
output_torch = ttnn.to_torch(output)
print(f"TTNN first 4: {output_torch[0,0,0,:4]}")
max_diff = (ref.float() - output_torch.float()).abs().max()
print(f"Max diff: {max_diff}")
ttnn.close_device(device)
