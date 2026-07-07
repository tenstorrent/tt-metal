import torch, ttnn, math

device = ttnn.open_device(device_id=0)

# MQA with H_q=71 — Falcon-7B config
q_shape = (1, 71, 2048, 64)
k_shape = (1, 1, 2048, 64)
v_shape = (1, 1, 2048, 64)

torch.manual_seed(42)
Q = torch.randn(q_shape, dtype=torch.bfloat16)
K = torch.randn(k_shape, dtype=torch.bfloat16)
V = torch.randn(v_shape, dtype=torch.bfloat16)

repeats = q_shape[1] // k_shape[1]
Kf = K.to(torch.float32).repeat_interleave(repeats, dim=1)
Vf = V.to(torch.float32).repeat_interleave(repeats, dim=1)
Qf = Q.to(torch.float32)
expected = torch.nn.functional.scaled_dot_product_attention(Qf, Kf, Vf).to(torch.bfloat16)

ttnn_Q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
ttnn_K = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
ttnn_V = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

try:
    result = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V)
    out = ttnn.to_torch(result)

    has_nan = torch.isnan(out).any().item()
    has_inf = torch.isinf(out).any().item()
    print(f"Has NaN: {has_nan}, Has Inf: {has_inf}")
    print(f"Output shape: {out.shape}")
    if not has_nan and not has_inf:

        def pcc(a, b):
            a = a.float().flatten()
            b = b.float().flatten()
            return torch.corrcoef(torch.stack([a, b]))[0, 1].item()

        print(f"PCC: {pcc(out, expected)}")
except Exception as e:
    print(f"Error: {e}")

ttnn.close_device(device)
