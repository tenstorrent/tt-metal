import torch, ttnn

device = ttnn.open_device(device_id=0)


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
        pcc_val = torch.corrcoef(torch.stack([out.float().flatten(), expected.float().flatten()]))[0, 1].item()
        max_diff = (out.float() - expected.float()).abs().max().item()
        status = "PASS" if pcc_val > 0.995 else "FAIL"
        print(f"{label}: PCC={pcc_val:.6f}, max_diff={max_diff:.6f} [{status}]")


# H_q=71 (MQA Falcon-7B) — 71 work units on 56 cores
test_gqa_mqa((1, 71, 2048, 64), (1, 1, 2048, 64), (1, 1, 2048, 64), "MQA H_q=71")

# H_q=64 (just over 56 cores)
test_gqa_mqa((1, 64, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64), "MQA H_q=64")

# Normal shapes (no regression check)
test_gqa_mqa((1, 8, 128, 64), (1, 2, 128, 64), (1, 2, 128, 64), "GQA 4:1")
test_gqa_mqa((1, 8, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64), "MQA 8:1")

ttnn.close_device(device)
