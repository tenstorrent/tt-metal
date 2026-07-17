import torch, ttnn, math
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

dev = ttnn.open_device(device_id=0)
try:

    def run(B, H, S, D, label, Vfn=None):
        Q = torch.ones(B, H, S, D, dtype=torch.bfloat16)
        K = torch.ones(B, H, S, D, dtype=torch.bfloat16)
        V = Vfn() if Vfn else torch.ones(B, H, S, D, dtype=torch.bfloat16)
        scale = 1.0 / math.sqrt(D)
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, math_approx_mode=False
        )
        tq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        tk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        tv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        out = scaled_dot_product_attention(tq, tk, tv, scale=scale, compute_kernel_config=cfg)
        r = ttnn.to_torch(out).float()
        exp = torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float(), scale=scale)
        maxdiff = (r - exp).abs().max().item()
        print(
            f"{label}: shape {tuple(r.shape)} min {r.min():.4f} max {r.max():.4f} mean {r.mean():.4f} maxdiff {maxdiff:.4f}"
        )

    run(1, 1, 64, 64, "allones 64x64")
    run(1, 1, 128, 128, "allones 128x128 (Dt=4 like flagged)")
    # random V to check l is really computed (not just uniform)
    torch.manual_seed(0)
    run(1, 1, 128, 128, "randV 128x128", Vfn=lambda: torch.randn(1, 1, 128, 128, dtype=torch.bfloat16))
finally:
    ttnn.close_device(dev)
