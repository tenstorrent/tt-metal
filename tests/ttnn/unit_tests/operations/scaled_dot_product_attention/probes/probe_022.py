import torch, ttnn

device = ttnn.open_device(device_id=0)

# Test S=1024 bf16 MHA (was passing before, now NaN)
q_shape = (1, 1, 1024, 64)
torch.manual_seed(42)
Q = torch.randn(q_shape, dtype=torch.bfloat16)
K = torch.randn(q_shape, dtype=torch.bfloat16)
V = torch.randn(q_shape, dtype=torch.bfloat16)

expected = torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float()).to(torch.bfloat16)

ttnn_Q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
ttnn_K = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
ttnn_V = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

result = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V)
out = ttnn.to_torch(result)

has_nan = torch.isnan(out).any().item()
has_inf = torch.isinf(out).any().item()
print(f"S=1024 D=64 bf16: NaN={has_nan} Inf={has_inf}")
if not has_nan and not has_inf:
    pcc = torch.corrcoef(torch.stack([out.float().flatten(), expected.float().flatten()]))[0, 1].item()
    print(f"PCC: {pcc}")

# Also test S=512 (should work)
q_shape2 = (1, 1, 512, 64)
Q2 = torch.randn(q_shape2, dtype=torch.bfloat16)
K2 = torch.randn(q_shape2, dtype=torch.bfloat16)
V2 = torch.randn(q_shape2, dtype=torch.bfloat16)
expected2 = torch.nn.functional.scaled_dot_product_attention(Q2.float(), K2.float(), V2.float()).to(torch.bfloat16)
ttnn_Q2 = ttnn.from_torch(Q2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
ttnn_K2 = ttnn.from_torch(K2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
ttnn_V2 = ttnn.from_torch(V2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
result2 = scaled_dot_product_attention(ttnn_Q2, ttnn_K2, ttnn_V2)
out2 = ttnn.to_torch(result2)
has_nan2 = torch.isnan(out2).any().item()
print(f"S=512 D=64 bf16: NaN={has_nan2}")
if not has_nan2:
    pcc2 = torch.corrcoef(torch.stack([out2.float().flatten(), expected2.float().flatten()]))[0, 1].item()
    print(f"PCC: {pcc2}")

ttnn.close_device(device)
