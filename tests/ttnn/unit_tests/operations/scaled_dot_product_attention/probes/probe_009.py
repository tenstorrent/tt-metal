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

out = ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention(tQ, tK, tV)
result = ttnn.to_torch(out)

# Where are the infs?
inf_mask = torch.isinf(result.float())
inf_count = inf_mask.sum().item()
print(f"Total inf count: {inf_count}")
print(f"Result shape: {list(result.shape)}")

# Check each row
for r in range(result.shape[2]):
    row = result.float()[0, 0, r, :]
    has_inf = torch.isinf(row).any().item()
    if has_inf:
        inf_cols = torch.isinf(row).nonzero(as_tuple=True)[0]
        print(
            f"  Row {r}: {len(inf_cols)} inf cols, first inf at col {inf_cols[0].item() if len(inf_cols) > 0 else -1}"
        )
        break

# Check what the first valid row looks like
print(f"Row 0 values: {result.float()[0,0,0,:].tolist()[:10]}")

# Now test WITHOUT the padding mask to see if it's the mask causing the issue
# Directly call the op bypassing _make_padding_mask
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
pd = create_program_descriptor(tQ, tK, tV, output_tensor, attn_mask=None, is_causal=False, scale=None)
out2 = ttnn.generic_op([tQ, tK, tV, output_tensor], pd)
result2 = ttnn.to_torch(out2)

Qf, Kf, Vf = Q.float(), K.float(), V.float()
scale = 1.0 / math.sqrt(64)
scores = Qf @ Kf.transpose(-1, -2) * scale
weights = torch.softmax(scores, dim=-1)
expected = (weights @ Vf).to(torch.bfloat16)

valid_diff2 = (result2.float()[:, :, :47, :] - expected.float()).abs()
print(f"Without mask - valid region max_diff: {valid_diff2.max().item()}")
print(f"Without mask - has inf: {torch.isinf(result2.float()).any().item()}")

ttnn.close_device(device)
