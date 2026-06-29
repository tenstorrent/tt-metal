import torch, ttnn, math

device = ttnn.open_device(device_id=0)

# Test: non-aligned S_q=47, D=64, with a TILE-ALIGNED mask (64x64)
torch.manual_seed(42)
Q = torch.randn(1, 1, 47, 64, dtype=torch.bfloat16)
K = torch.randn(1, 1, 47, 64, dtype=torch.bfloat16)
V = torch.randn(1, 1, 47, 64, dtype=torch.bfloat16)

# Create a TILE-ALIGNED mask: (1, 1, 64, 64) with -inf in cols 47-63
mask = torch.zeros(1, 1, 64, 64, dtype=torch.bfloat16)
mask[:, :, :, 47:] = float("-inf")

tQ = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tK = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tV = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tM = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

print(f"tM shape: {list(tM.shape)}")
print(f"tQ shape: {list(tQ.shape)}")
print(f"tK shape: {list(tK.shape)}")

out = ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention(tQ, tK, tV, attn_mask=tM)
result = ttnn.to_torch(out)

Qf, Kf, Vf = Q.float(), K.float(), V.float()
scale = 1.0 / math.sqrt(64)
scores = Qf @ Kf.transpose(-1, -2) * scale + mask[:, :, :47, :].float()
weights = torch.softmax(scores, dim=-1)
expected = (weights @ Vf).to(torch.bfloat16)

valid_diff = (result.float()[:, :, :47, :] - expected.float()).abs().max().item()
print(f"Tile-aligned mask + non-aligned S_q: valid max_diff={valid_diff:.6f}")
print(f"Has inf: {torch.isinf(result.float()).any().item()}")
print(f"Result[0,0,0,:5] = {result.float()[0,0,0,:5].tolist()}")

ttnn.close_device(device)
