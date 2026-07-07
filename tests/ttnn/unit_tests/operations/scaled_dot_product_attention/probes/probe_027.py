import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device_id = 0
device = ttnn.CreateDevice(device_id)

torch.manual_seed(42)
B, H, S_q, D, S_kv = 1, 1, 1024, 64, 1024
q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16)
k = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
v = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)

ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)

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
result = ttnn.to_torch(output)

print(f"Output shape: {result.shape}")
print(f"Output min: {result.float().min()}, max: {result.float().max()}")
print(f"Ref min: {ref.float().min()}, max: {ref.float().max()}")
print(f"Output[0,0,:5,:5]:\n{result.float()[0,0,:5,:5]}")
print(f"Ref[0,0,:5,:5]:\n{ref.float()[0,0,:5,:5]}")
print(f"Any NaN: {result.float().isnan().any()}")
print(f"Any Inf: {result.float().isinf().any()}")
print(f"Max diff: {(result.float() - ref.float()).abs().max()}")
print(f"Output abs sum: {result.float().abs().sum()}")

ttnn.CloseDevice(device)
