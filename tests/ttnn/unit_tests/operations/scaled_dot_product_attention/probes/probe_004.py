import torch, ttnn, math

device = ttnn.open_device(device_id=0)

# Test 1: D non-aligned (w_non_aligned)
torch.manual_seed(42)
Q = torch.randn(1, 1, 32, 50, dtype=torch.bfloat16)
K = torch.randn(1, 1, 32, 50, dtype=torch.bfloat16)
V = torch.randn(1, 1, 32, 50, dtype=torch.bfloat16)

Qf, Kf, Vf = Q.float(), K.float(), V.float()
scale = 1.0 / math.sqrt(50)
scores = Qf @ Kf.transpose(-1, -2) * scale
weights = torch.softmax(scores, dim=-1)
expected = (weights @ Vf).to(torch.bfloat16)

tQ = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tK = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tV = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
out = ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention(tQ, tK, tV)
result = ttnn.to_torch(out)

max_diff = (result.float() - expected.float()).abs().max().item()
print(f"Test 1 (D non-aligned, 32x50): max_diff={max_diff:.6f}, shape={list(result.shape)}")

ttnn.close_device(device)
