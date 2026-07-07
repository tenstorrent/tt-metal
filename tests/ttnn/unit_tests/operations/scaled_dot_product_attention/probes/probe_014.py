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

    pcc_val = pcc(out, expected)
    max_diff = (out.float() - expected.float()).abs().max().item()
    status = "PASS" if pcc_val > 0.995 else "FAIL"
    print(f"{label}: PCC={pcc_val:.6f}, max_diff={max_diff:.6f} [{status}]")


# MQA shapes
test_gqa_mqa((1, 8, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64), "MQA 8:1")
test_gqa_mqa((1, 12, 128, 64), (1, 1, 128, 64), (1, 1, 128, 64), "MQA 12:1")
test_gqa_mqa((1, 32, 128, 128), (1, 1, 128, 128), (1, 1, 128, 128), "MQA 32:1")

# GQA shapes
test_gqa_mqa((1, 8, 256, 64), (1, 2, 256, 64), (1, 2, 256, 64), "GQA 4:1 long")
test_gqa_mqa((1, 32, 128, 128), (1, 8, 128, 128), (1, 8, 128, 128), "GQA Llama3 4:1")
test_gqa_mqa((1, 12, 128, 64), (1, 4, 128, 64), (1, 4, 128, 64), "GQA 3:1")
test_gqa_mqa((1, 16, 256, 64), (1, 4, 256, 64), (1, 4, 256, 64), "GQA 4:1 GPT")

# MQA cross-attention
test_gqa_mqa((1, 8, 64, 64), (1, 1, 128, 64), (1, 1, 128, 64), "MQA cross")

# GQA cross-attention
test_gqa_mqa((1, 8, 64, 64), (1, 2, 128, 64), (1, 2, 128, 64), "GQA cross")

# Multi-batch
test_gqa_mqa((2, 8, 128, 64), (2, 2, 128, 64), (2, 2, 128, 64), "GQA 4:1 batch")
test_gqa_mqa((2, 8, 128, 64), (2, 1, 128, 64), (2, 1, 128, 64), "MQA batch")

ttnn.close_device(device)
