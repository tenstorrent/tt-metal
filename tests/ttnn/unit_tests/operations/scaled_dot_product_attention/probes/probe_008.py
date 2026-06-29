import torch, ttnn, math

device = ttnn.open_device(device_id=0)

# S non-aligned: S_q=S_kv=47, D=64
torch.manual_seed(42)
Q = torch.randn(1, 1, 47, 64, dtype=torch.bfloat16)
K = torch.randn(1, 1, 47, 64, dtype=torch.bfloat16)
V = torch.randn(1, 1, 47, 64, dtype=torch.bfloat16)

tQ = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tK = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tV = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

# Check the output without the padding mask (directly)
# First check what the program descriptor sees
from ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention import _make_padding_mask

effective_mask, is_per_head = _make_padding_mask(tQ, tK, None, is_causal=False)
print(f"mask shape: {list(effective_mask.shape)}")
print(f"mask dtype: {effective_mask.dtype}")
# Check physical shape (padded to tiles)
print(f"mask shape with padding: {list(ttnn.to_torch(effective_mask).shape)}")

# Check what the program descriptor computes for S_q_tiles, S_kv_tiles
# S_q=47 -> S_q_tiles = ceil(47/32) = 2
# S_kv=47 -> S_kv_tiles = ceil(47/32) = 2
# D=64 -> D_t = 2
# B_q_t = min(4, 2) = 2, B_kv_t = min(4, 2) = 2
# num_q_blocks = ceil(2/2) = 1
# num_kv_blocks = ceil(2/2) = 1
# num_score_tiles = 2*2 = 4
# num_o_tiles = 2*2 = 4

# Now run the op
out = ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention(tQ, tK, tV)
result = ttnn.to_torch(out)

# Check valid region only
Qf, Kf, Vf = Q.float(), K.float(), V.float()
scale = 1.0 / math.sqrt(64)
scores = Qf @ Kf.transpose(-1, -2) * scale
weights = torch.softmax(scores, dim=-1)
expected = (weights @ Vf).to(torch.bfloat16)

valid_diff = (result.float()[:, :, :47, :] - expected.float()).abs()
print(f"Valid region max_diff: {valid_diff.max().item()}")
print(f"Result[0,0,0,:8] = {result.float()[0,0,0,:8].tolist()}")
print(f"Expected[0,0,0,:8] = {expected.float()[0,0,0,:8].tolist()}")
# Check if result is all zero
print(f"Result all zero (valid): {(result.float()[:,:,:47,:64] == 0).all().item()}")
print(f"Result has inf: {torch.isinf(result.float()).any().item()}")

ttnn.close_device(device)
