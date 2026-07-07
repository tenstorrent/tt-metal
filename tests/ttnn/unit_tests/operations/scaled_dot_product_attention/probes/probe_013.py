import torch, ttnn, math

device = ttnn.open_device(device_id=0)

# GQA 4:1 ratio
q_shape = (1, 8, 128, 64)
k_shape = (1, 2, 128, 64)
v_shape = (1, 2, 128, 64)

torch.manual_seed(42)
Q = torch.randn(q_shape, dtype=torch.bfloat16)
K = torch.randn(k_shape, dtype=torch.bfloat16)
V = torch.randn(v_shape, dtype=torch.bfloat16)

# Reference: repeat_interleave K/V heads
repeats = q_shape[1] // k_shape[1]
Kf = K.to(torch.float32).repeat_interleave(repeats, dim=1)
Vf = V.to(torch.float32).repeat_interleave(repeats, dim=1)
Qf = Q.to(torch.float32)
expected = torch.nn.functional.scaled_dot_product_attention(Qf, Kf, Vf)
expected = expected.to(torch.bfloat16)

ttnn_Q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
ttnn_K = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
ttnn_V = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

result = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V)
out = ttnn.to_torch(result)


# Compute PCC
def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


pcc_val = pcc(out, expected)
max_diff = (out.float() - expected.float()).abs().max().item()
print(f"GQA 4:1 — PCC={pcc_val:.6f}, max_diff={max_diff:.6f}")
print(f"Output shape: {out.shape}")
print(f"Expected shape: {expected.shape}")

ttnn.close_device(device)
