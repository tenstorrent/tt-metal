import torch, ttnn, math

device = ttnn.open_device(device_id=0)

# Simpler test: S_q=47, S_kv=47, D=64, with a mask that has shape (1,1,64,64) (tile-aligned)
# This avoids the _make_padding_mask issue
torch.manual_seed(42)
Q = torch.randn(1, 1, 47, 64, dtype=torch.bfloat16)
K = torch.randn(1, 1, 47, 64, dtype=torch.bfloat16)
V = torch.randn(1, 1, 47, 64, dtype=torch.bfloat16)

# Create mask with shape (1,1,64,64) — tile-aligned in both dims
# -inf in cols 47-63 (padding S_kv)
mask = torch.zeros(1, 1, 64, 64, dtype=torch.bfloat16)
mask[:, :, :, 47:] = float("-inf")

tQ = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tK = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tV = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tM = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

# Bypass _make_padding_mask by calling program descriptor directly
from ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention_program_descriptor import (
    create_program_descriptor,
)

output_tensor = ttnn.allocate_tensor_on_device(
    ttnn.Shape(list(tQ.shape)),
    tQ.dtype,
    ttnn.TILE_LAYOUT,
    device,
    ttnn.DRAM_MEMORY_CONFIG,
)
pd = create_program_descriptor(tQ, tK, tV, output_tensor, attn_mask=tM, is_causal=False, scale=None)
out = ttnn.generic_op([tQ, tK, tV, output_tensor, tM], pd)
result = ttnn.to_torch(out)

Qf, Kf, Vf = Q.float(), K.float(), V.float()
scale = 1.0 / math.sqrt(64)
scores = Qf @ Kf.transpose(-1, -2) * scale + mask[:, :, :47, :47].float()
weights = torch.softmax(scores, dim=-1)
expected = (weights @ Vf).to(torch.bfloat16)

valid_diff = (result.float()[:, :, :47, :] - expected.float()).abs().max().item()
print(f"Direct tile-aligned mask + non-aligned S_q: valid max_diff={valid_diff:.6f}")
print(f"Has inf: {torch.isinf(result.float()).any().item()}")
print(f"Result[0,0,0,:5] = {result.float()[0,0,0,:5].tolist()}")
print(f"Expected[0,0,0,:5] = {expected.float()[0,0,0,:5].tolist()}")

ttnn.close_device(device)
