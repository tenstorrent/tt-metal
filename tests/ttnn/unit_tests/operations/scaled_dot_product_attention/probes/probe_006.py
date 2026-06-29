import torch, ttnn, math

device = ttnn.open_device(device_id=0)

# Test 2: S non-aligned (h_non_aligned)
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

# Check valid region only (rows 0-46)
valid_diff = (result.float()[:, :, :47, :] - expected.float()).abs()
print(f"Valid region (rows 0-46): max_diff={valid_diff.max().item():.6f}")
print(f"Result has inf: {torch.isinf(result.float()).any().item()}")
print(f"Result has nan: {torch.isnan(result.float()).any().item()}")
if torch.isinf(result.float()).any() or torch.isnan(result.float()).any():
    inf_mask = torch.isinf(result.float()) | torch.isnan(result.float())
    inf_rows = inf_mask.any(dim=-1).any(dim=-1).any(dim=0).nonzero(as_tuple=True)[0]
    print(f"Inf/nan rows: {inf_rows.tolist()}")
    # Check a valid row
    print(f"Row 0 max_diff: {(result.float()[0,0,0,:] - expected.float()[0,0,0,:]).abs().max().item():.6f}")
    print(f"Row 46 max_diff: {(result.float()[0,0,46,:] - expected.float()[0,0,46,:]).abs().max().item():.6f}")

ttnn.close_device(device)
