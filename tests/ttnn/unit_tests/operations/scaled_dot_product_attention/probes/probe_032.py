import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.CreateDevice(0)

torch.manual_seed(0)
B, H, S, D = 1, 1, 128, 256
q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
k = torch.randn(B, H, S, D, dtype=torch.bfloat16)
v = torch.randn(B, H, S, D, dtype=torch.bfloat16)

ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)

q_t = ttnn.from_torch(
    q, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
k_t = ttnn.from_torch(
    k, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
v_t = ttnn.from_torch(
    v, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)

ckc = ttnn.ComputeConfigDescriptor(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=False,
    math_approx_mode=False,
)

output = scaled_dot_product_attention(q_t, k_t, v_t, compute_kernel_config=ckc)
result = ttnn.to_torch(output)

print(f"Output min: {result.float().min()}, max: {result.float().max()}")
print(f"Ref min: {ref.float().min()}, max: {ref.float().max()}")
print(f"nan: {result.float().isnan().any()}, inf: {result.float().isinf().any()}")
print(f"Output[0,0,0,:5]: {result.float()[0,0,0,:5]}")
print(f"Ref[0,0,0,:5]: {ref.float()[0,0,0,:5]}")

ttnn.CloseDevice(device)
