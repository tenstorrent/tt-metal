import torch, ttnn, math

device = ttnn.open_device(device_id=0)


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def test_gqa_mqa(q_shape, k_shape, v_shape, label):
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

    result = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V)
    out = ttnn.to_torch(result)

    has_nan = torch.isnan(out).any().item()
    has_inf = torch.isinf(out).any().item()
    if has_nan or has_inf:
        print(f"{label}: NaN={has_nan} Inf={has_inf} [FAIL]")
    else:
        pcc_val = pcc(out, expected)
        max_diff = (out.float() - expected.float()).abs().max().item()
        status = "PASS" if pcc_val > 0.995 else "FAIL"
        print(f"{label}: PCC={pcc_val:.6f}, max_diff={max_diff:.6f} [{status}]")


# The problematic H_q=71 shape
test_gqa_mqa((1, 71, 2048, 64), (1, 1, 2048, 64), (1, 1, 2048, 64), "MQA H_q=71")

# Also test a shape that needs >56 work units but <71
test_gqa_mqa((1, 64, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64), "MQA H_q=64")

# And a simple multi-batch shape that needs >56 work units
test_gqa_mqa((8, 8, 128, 64), (8, 1, 128, 64), (8, 1, 128, 64), "MQA B=8 H_q=8 (64 units)")

# And a normal MHA shape to check no regression
test_gqa_mqa((1, 8, 128, 64), (1, 8, 128, 64), (1, 8, 128, 64), "MHA H=8")

ttnn.close_device(device)
