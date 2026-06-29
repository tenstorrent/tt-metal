import torch, ttnn, math

device = ttnn.open_device(device_id=0)

# Test 1: D non-aligned (w_non_aligned)
# Shape: (1, 1, 32, 50) - S=32 (aligned), D=50 (non-aligned)
torch.manual_seed(42)
Q = torch.randn(1, 1, 32, 50, dtype=torch.bfloat16)
K = torch.randn(1, 1, 32, 50, dtype=torch.bfloat16)
V = torch.randn(1, 1, 32, 50, dtype=torch.bfloat16)

# PyTorch reference
Qf, Kf, Vf = Q.float(), K.float(), V.float()
scale = 1.0 / math.sqrt(50)
scores = Qf @ Kf.transpose(-1, -2) * scale
weights = torch.softmax(scores, dim=-1)
expected = (weights @ Vf).to(torch.bfloat16)

# TTNN
tQ = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tK = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tV = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
out = ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention(tQ, tK, tV)
result = ttnn.to_torch(out)

# Check
max_diff = (result.float() - expected.float()).abs().max().item()
print(f"Test 1 (D non-aligned, 32x50): max_diff={max_diff:.6f}, shape={list(result.shape)}")

# Test 2: S non-aligned (h_non_aligned)
# Shape: (1, 1, 47, 64) - S=47 (non-aligned), D=64 (aligned)
Q2 = torch.randn(1, 1, 47, 64, dtype=torch.bfloat16)
K2 = torch.randn(1, 1, 47, 64, dtype=torch.bfloat16)
V2 = torch.randn(1, 1, 47, 64, dtype=torch.bfloat16)

Qf2, Kf2, Vf2 = Q2.float(), K2.float(), V2.float()
scale2 = 1.0 / math.sqrt(64)
scores2 = Qf2 @ Kf2.transpose(-1, -2) * scale2
weights2 = torch.softmax(scores2, dim=-1)
expected2 = (weights2 @ Vf2).to(torch.bfloat16)

tQ2 = ttnn.from_torch(Q2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tK2 = ttnn.from_torch(K2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tV2 = ttnn.from_torch(V2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
out2 = ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention(tQ2, tK2, tV2)
result2 = ttnn.to_torch(out2)

max_diff2 = (result2.float() - expected2.float()).abs().max().item()
print(f"Test 2 (S non-aligned, 47x64): max_diff={max_diff2:.6f}, shape={list(result2.shape)}")

# Test 3: Both non-aligned
# Shape: (1, 1, 47, 50) - S=47, D=50
Q3 = torch.randn(1, 1, 47, 50, dtype=torch.bfloat16)
K3 = torch.randn(1, 1, 47, 50, dtype=torch.bfloat16)
V3 = torch.randn(1, 1, 47, 50, dtype=torch.bfloat16)

Qf3, Kf3, Vf3 = Q3.float(), K3.float(), V3.float()
scale3 = 1.0 / math.sqrt(50)
scores3 = Qf3 @ Kf3.transpose(-1, -2) * scale3
weights3 = torch.softmax(scores3, dim=-1)
expected3 = (weights3 @ Vf3).to(torch.bfloat16)

tQ3 = ttnn.from_torch(Q3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tK3 = ttnn.from_torch(K3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tV3 = ttnn.from_torch(V3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
out3 = ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention(tQ3, tK3, tV3)
result3 = ttnn.to_torch(out3)

max_diff3 = (result3.float() - expected3.float()).abs().max().item()
print(f"Test 3 (both non-aligned, 47x50): max_diff={max_diff3:.6f}, shape={list(result3.shape)}")

ttnn.close_device(device)
