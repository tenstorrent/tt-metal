import torch, ttnn, math

device = ttnn.open_device(device_id=0)

# Test: tile-aligned shape WITH explicit mask to verify mask path works
torch.manual_seed(42)
Q = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)  # tile-aligned
K = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
V = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)

# Create a mask: (1, 1, 64, 64) with -inf in cols 47-63 (same pattern as padding mask)
mask = torch.zeros(1, 1, 64, 64, dtype=torch.bfloat16)
mask[:, :, :, 47:] = float("-inf")

tQ = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tK = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tV = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tM = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

out = ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention(tQ, tK, tV, attn_mask=tM)
result = ttnn.to_torch(out)

# Reference with mask
Qf, Kf, Vf = Q.float(), K.float(), V.float()
scale = 1.0 / math.sqrt(64)
scores = Qf @ Kf.transpose(-1, -2) * scale + mask.float()
weights = torch.softmax(scores, dim=-1)
expected = (weights @ Vf).to(torch.bfloat16)

max_diff = (result.float() - expected.float()).abs().max().item()
print(f"Tile-aligned with mask (64x64, -inf in cols 47-63): max_diff={max_diff:.6f}")
print(f"Has inf: {torch.isinf(result.float()).any().item()}")
print(f"Has nan: {torch.isnan(result.float()).any().item()}")

ttnn.close_device(device)
