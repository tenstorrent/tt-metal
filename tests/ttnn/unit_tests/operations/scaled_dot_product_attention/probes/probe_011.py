import torch, math, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.open_device(device_id=0)


def vones(S):
    sh = (1, 1, S, 64)
    torch.manual_seed(0)
    Q = torch.randn(sh, dtype=torch.bfloat16)
    K = torch.randn(sh, dtype=torch.bfloat16)
    V = torch.ones(sh, dtype=torch.bfloat16)
    tq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(scaled_dot_product_attention(tq, tk, tv)).float()[0, 0]
    err = out - 1.0
    row_mean = err.mean(-1)
    row_std = err.std(-1)
    print(
        f"PROBE S={S}: max={err.abs().max():.5f} rowmean_absmax={row_mean.abs().max():.5f} "
        f"rowstd_max={row_std.max():.6f} frac_rows_gt_0.01={(row_mean.abs() > 0.01).float().mean():.3f}"
    )


for S in (128, 256, 512, 1024, 4096):
    vones(S)
ttnn.close_device(device)
