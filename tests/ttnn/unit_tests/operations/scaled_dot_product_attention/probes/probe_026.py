import torch, ttnn, math
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.open_device(device_id=0)
# Shape (1,1,128,64): num_q_blocks=1, num_kv_blocks=2 → PASSES
# Shape (1,1,256,64): num_q_blocks=2, num_kv_blocks=2 → FAILS
# Test (1,1,256,64) but compare per-Q-block output with (1,1,128,64)

torch.manual_seed(42)
# First Q-block of 256x64 = first 128 rows
Q1 = torch.randn(1, 1, 128, 64, dtype=torch.bfloat16)
K = torch.randn(1, 1, 256, 64, dtype=torch.bfloat16)
V = torch.randn(1, 1, 256, 64, dtype=torch.bfloat16)

# Full 256x64
Q_full = torch.randn(1, 1, 256, 64, dtype=torch.bfloat16)
# Use same seed but different shape — actually need same first 128 rows
# Let's use Q_full[:,:,:128,:] as Q1
torch.manual_seed(42)
Q_full = torch.randn(1, 1, 256, 64, dtype=torch.bfloat16)
Q1 = Q_full[:, :, :128, :].contiguous()

scale = 1.0 / math.sqrt(64)
Qf, Kf, Vf = Q_full.float(), K.float(), V.float()
scores = (Qf @ Kf.transpose(-1, -2)) * scale
weights = torch.softmax(scores, dim=-1)
expected_full = (weights @ Vf).to(torch.bfloat16)

# Also compute expected for just Q1
Q1f = Q1.float()
scores1 = (Q1f @ Kf.transpose(-1, -2)) * scale
weights1 = torch.softmax(scores1, dim=-1)
expected1 = (weights1 @ Vf).to(torch.bfloat16)

ttnn_Q_full = ttnn.from_torch(
    Q_full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
ttnn_Q1 = ttnn.from_torch(
    Q1, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
ttnn_K = ttnn.from_torch(
    K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
ttnn_V = ttnn.from_torch(
    V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)

# Run 128x64 (num_q_blocks=1, num_kv_blocks=2)
out1 = scaled_dot_product_attention(ttnn_Q1, ttnn_K, ttnn_V)
res1 = ttnn.to_torch(out1)

# Run 256x64 (num_q_blocks=2, num_kv_blocks=2)
out_full = scaled_dot_product_attention(ttnn_Q_full, ttnn_K, ttnn_V)
res_full = ttnn.to_torch(out_full)

# Compare first Q-block of full output with 128x64 output
diff_qb0 = (res_full[0, 0, :128, :].float() - res1[0, 0, :, :].float()).abs().max()
print(f"Q-block 0: 256x64 vs 128x64 max diff: {diff_qb0:.6f}")

# Compare first Q-block of full output with expected
diff_qb0_exp = (res_full[0, 0, :128, :].float() - expected1[0, 0, :, :].float()).abs().max()
print(f"Q-block 0: 256x64 vs expected max diff: {diff_qb0_exp:.6f}")

# Compare 128x64 output with expected
diff_128_exp = (res1[0, 0, :, :].float() - expected1[0, 0, :, :].float()).abs().max()
print(f"128x64 vs expected max diff: {diff_128_exp:.6f}")

ttnn.close_device(device)
