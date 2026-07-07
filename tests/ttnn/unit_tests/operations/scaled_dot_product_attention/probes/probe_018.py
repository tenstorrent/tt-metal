import torch, ttnn

device = ttnn.open_device(device_id=0)

# Simple MHA test
q_shape = (1, 8, 128, 64)
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
print(f"NaN: {has_nan}")
if not has_nan:
    pcc = torch.corrcoef(torch.stack([out.float().flatten(), expected.float().flatten()]))[0, 1].item()
    print(f"PCC: {pcc}")

ttnn.close_device(device)
