import torch, ttnn, math

device = ttnn.open_device(device_id=0)


def run_test(label, Q_shape, K_shape, V_shape, use_mask=False, scale_val=None):
    torch.manual_seed(42)
    Q = torch.randn(Q_shape, dtype=torch.bfloat16)
    K = torch.randn(K_shape, dtype=torch.bfloat16)
    V = torch.randn(V_shape, dtype=torch.bfloat16)

    if use_mask:
        B, H, S_q, _D = Q_shape
        S_kv = K_shape[-2]
        mask = torch.zeros(B, 1, S_q, S_kv, dtype=torch.bfloat16)
        mask.masked_fill_(torch.triu(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=1), float("-inf"))
    else:
        mask = None

    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    D = Q_shape[-1]
    scale = scale_val if scale_val is not None else 1.0 / math.sqrt(D)

    # GQA/MQA: replicate K/V heads
    H_q, H_kv = Q_shape[1], K_shape[1]
    if H_q != H_kv:
        repeats = H_q // H_kv
        Kf = Kf.repeat_interleave(repeats, dim=1)
        Vf = Vf.repeat_interleave(repeats, dim=1)

    scores = Qf @ Kf.transpose(-1, -2) * scale
    if mask is not None:
        scores = scores + mask.float()
    weights = torch.softmax(scores, dim=-1)
    expected = (weights @ Vf).to(torch.bfloat16)

    tQ = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tK = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tV = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tM = (
        ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) if mask is not None else None
    )

    out = ttnn.operations.scaled_dot_product_attention.scaled_dot_product_attention(
        tQ, tK, tV, attn_mask=tM, scale=scale_val
    )
    result = ttnn.to_torch(out)

    # Compare valid region
    S_q = Q_shape[2]
    D = Q_shape[3]
    valid_diff = (result.float()[:, :, :S_q, :D] - expected.float()).abs().max().item()
    has_inf = torch.isinf(result.float()).any().item()
    has_nan = torch.isnan(result.float()).any().item()
    print(f"{label}: max_diff={valid_diff:.6f}, inf={has_inf}, nan={has_nan}")


# Test all non-aligned categories
run_test("D non-aligned (32x50)", (1, 1, 32, 50), (1, 1, 32, 50), (1, 1, 32, 50))
run_test("S non-aligned (47x64)", (1, 1, 47, 64), (1, 1, 47, 64), (1, 1, 47, 64))
run_test("Both non-aligned (47x50)", (1, 1, 47, 50), (1, 1, 47, 50), (1, 1, 47, 50))
run_test("D non-aligned + multi-head (8,64,47)", (1, 8, 64, 47), (1, 8, 64, 47), (1, 8, 64, 47))
run_test("S non-aligned + multi-head (4,47,64)", (1, 4, 47, 64), (1, 4, 47, 64), (1, 4, 47, 64))
run_test("S non-aligned + mask (47x64)", (1, 1, 47, 64), (1, 1, 47, 64), (1, 1, 47, 64), use_mask=True)
run_test("S non-aligned + GQA (8,47,64)", (1, 8, 47, 64), (1, 2, 47, 64), (1, 2, 47, 64))
run_test("S non-aligned + MQA (8,47,64)", (1, 8, 47, 64), (1, 1, 47, 64), (1, 1, 47, 64))
run_test("S non-aligned + multi-batch (2,4,100,64)", (2, 4, 100, 64), (2, 4, 100, 64), (2, 4, 100, 64))
run_test("Both non-aligned + multi-head (12,33,50)", (1, 12, 33, 50), (1, 12, 33, 50), (1, 12, 33, 50))
run_test("Cross-attn both non-aligned (4,100,50)", (1, 4, 100, 50), (1, 4, 47, 50), (1, 4, 47, 50))

ttnn.close_device(device)
