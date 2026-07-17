import torch, ttnn, math

torch.manual_seed(0)
dev = ttnn.open_device(device_id=0)
try:
    from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
    )

    def pcc(a, b):
        a = a.flatten().float()
        b = b.flatten().float()
        return float(((a - a.mean()) * (b - b.mean())).sum() / ((a - a.mean()).norm() * (b - b.mean()).norm() + 1e-12))

    for name, (B, H, S, D) in [("causal_512", (1, 2, 512, 64)), ("causal_1024_d128", (1, 4, 1024, 128))]:
        Q, K, V = (torch.randn(B, H, S, D) for _ in range(3))
        scale = 1.0 / math.sqrt(D)
        ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True, scale=scale)
        tq, tk, tv = (ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev) for x in (Q, K, V))
        out = scaled_dot_product_attention(tq, tk, tv, is_causal=True, scale=scale, compute_kernel_config=cfg)
        g = ttnn.to_torch(out).float()
        p = pcc(ref, g)
        md = float((g - ref).abs().max())
        print(
            f"{name} (fp32_dest=False, THROUGHPUT/fused super-fast-exp+ReLU): PCC={p:.5f} maxdiff={md:.4g} {'*** GARBAGE ***' if p<0.99 else 'OK'}"
        )
finally:
    ttnn.close_device(dev)
