import torch, ttnn, math

device = ttnn.open_device(device_id=0)

# Test: tile-aligned 64x64 with a mask that has -inf in ENTIRE last Q rows (rows 47-63)
# This simulates what happens when padded Q rows have all -inf scores
torch.manual_seed(42)
Q = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
K = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
V = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)

# Mask: -inf in rows 47-63 (all cols) AND -inf in cols 47-63 (all rows)
mask = torch.zeros(1, 1, 64, 64, dtype=torch.bfloat16)
mask[:, :, 47:, :] = float("-inf")  # padded Q rows
mask[:, :, :, 47:] = float("-inf")  # padded S_kv cols

tQ = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tK = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tV = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tM = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

out = ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention(tQ, tK, tV, attn_mask=tM)
result = ttnn.to_torch(out)

# Reference
Qf, Kf, Vf = Q.float(), K.float(), V.float()
scale = 1.0 / math.sqrt(64)
scores = Qf @ Kf.transpose(-1, -2) * scale + mask.float()
weights = torch.softmax(scores, dim=-1)
expected = (weights @ Vf).to(torch.bfloat16)

# Check valid region (rows 0-46, cols 0-46)
valid_diff = (result.float()[:, :47, :47] - expected.float()[:, :47, :47]).abs().max().item()
print(f"Valid region max_diff: {valid_diff:.6f}")
print(f"Result[:47,:47] has inf: {torch.isinf(result.float()[:,:,:47,:47]).any().item()}")
print(f"Result[:47,:47] has nan: {torch.isnan(result.float()[:,:,:47,:47]).any().item()}")
print(f"Result[47:,:] has inf: {torch.isinf(result.float()[:,:,47:,:]).any().item()}")
print(f"Result[47:,:] has nan: {torch.isnan(result.float()[:,:,47:,:]).any().item()}")

ttnn.close_device(device)
