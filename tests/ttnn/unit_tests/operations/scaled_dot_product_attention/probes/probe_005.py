import torch, ttnn, math

device = ttnn.open_device(device_id=0)

# Test 2: S non-aligned (h_non_aligned)
# Shape: (1, 1, 47, 64) - S=47 (non-aligned), D=64 (aligned)
torch.manual_seed(42)
Q = torch.randn(1, 1, 47, 64, dtype=torch.bfloat16)
K = torch.randn(1, 1, 47, 64, dtype=torch.bfloat16)
V = torch.randn(1, 1, 47, 64, dtype=torch.bfloat16)

Qf, Kf, Vf = Q.float(), K.float(), V.float()
scale = 1.0 / math.sqrt(64)
scores = Qf @ Kf.transpose(-1, -2) * scale
weights = torch.softmax(scores, dim=-1)
expected = (weights @ Vf).to(torch.bfloat16)

tQ = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tK = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tV = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
out = ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention(tQ, tK, tV)
result = ttnn.to_torch(out)

max_diff = (result.float() - expected.float()).abs().max().item()
# Also check PCC
out_flat = result.float().flatten()
exp_flat = expected.float().flatten()
out_c = out_flat - out_flat.mean()
exp_c = exp_flat - exp_flat.mean()
num = (out_c * exp_c).sum()
den = torch.sqrt((out_c**2).sum()) * torch.sqrt((exp_c**2).sum())
pcc = (num / den).item() if den > 0 else 1.0
print(f"Test 2 (S non-aligned, 47x64): max_diff={max_diff:.6f}, pcc={pcc:.6f}, shape={list(result.shape)}")

ttnn.close_device(device)
