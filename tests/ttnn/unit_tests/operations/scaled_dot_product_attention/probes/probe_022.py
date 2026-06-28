import torch, ttnn, math
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.open_device(device_id=0)
# Shape (1,1,256,64): num_q_blocks=2, num_kv_blocks=2
# Use simple deterministic input: all 0.5
torch.manual_seed(42)
Q = torch.full((1, 1, 256, 64), 0.5, dtype=torch.bfloat16)
K = torch.full((1, 1, 256, 64), 0.5, dtype=torch.bfloat16)
V = torch.full((1, 1, 256, 64), 0.5, dtype=torch.bfloat16)

scale = 1.0 / math.sqrt(64)
Qf, Kf, Vf = Q.float(), K.float(), V.float()
scores = (Qf @ Kf.transpose(-1, -2)) * scale  # each element = 0.5*0.5*64 * scale = 16/sqrt(64) = 2.0
weights = torch.softmax(scores, dim=-1)  # uniform = 1/256
expected = (weights @ Vf).to(torch.bfloat16)  # 0.5 * 1 = 0.5
print(f"Expected: all 0.5, actual first 4: {expected[0,0,0,:4]}")

ttnn_Q = ttnn.from_torch(
    Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
ttnn_K = ttnn.from_torch(
    K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
ttnn_V = ttnn.from_torch(
    V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)

output = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V)
result = ttnn.to_torch(output)
print(f"Result first 4x4:\n{result[0,0,:4,:4]}")
print(f"Expected first 4x4:\n{expected[0,0,:4,:4]}")
print(f"Max diff: {(result.float() - expected.float()).abs().max():.6f}")
# Check each Q-block separately
print(f"Q-block 0 (rows 0-127) max diff: {(result[0,0,:128,:].float() - expected[0,0,:128,:].float()).abs().max():.6f}")
print(
    f"Q-block 1 (rows 128-255) max diff: {(result[0,0,128:,:].float() - expected[0,0,128:,:].float()).abs().max():.6f}"
)

ttnn.close_device(device)
