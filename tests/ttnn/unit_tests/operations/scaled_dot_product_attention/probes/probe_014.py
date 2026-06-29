import torch, ttnn, math

device = ttnn.open_device(device_id=0)

# Test: S non-aligned (47x64) - no user mask
torch.manual_seed(42)
Q = torch.randn(1, 1, 47, 64, dtype=torch.bfloat16)
K = torch.randn(1, 1, 47, 64, dtype=torch.bfloat16)
V = torch.randn(1, 1, 47, 64, dtype=torch.bfloat16)

tQ = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tK = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tV = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
out = ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention(tQ, tK, tV)
result = ttnn.to_torch(out)

Qf, Kf, Vf = Q.float(), K.float(), V.float()
scale = 1.0 / math.sqrt(64)
scores = Qf @ Kf.transpose(-1, -2) * scale
weights = torch.softmax(scores, dim=-1)
expected = (weights @ Vf).to(torch.bfloat16)

valid_diff = (result.float()[:, :, :47, :] - expected.float()).abs().max().item()
has_inf = torch.isinf(result.float()).any().item()
print(f"S non-aligned 47x64: max_diff={valid_diff:.6f}, has_inf={has_inf}")
print(f"Result[0,0,0,:5] = {result.float()[0,0,0,:5].tolist()}")

ttnn.close_device(device)
