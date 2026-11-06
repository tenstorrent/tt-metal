import torch, ttnn
from ttnn.operations.mhc_post import mhc_post

N = 4
NG = 24
device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    T, C = 32, 64
    f = torch.randn(1, 1, T, C).bfloat16()
    x = torch.randn(1, 1, T, N * C).bfloat16()
    post = (2 * torch.sigmoid(torch.randn(1, 1, T, N))).float()
    comb = torch.softmax(torch.randn(1, 1, T, N * N), -1).float()
    nc = N * C
    mix = (
        post.double().reshape(T, N, 1) * f.double().reshape(T, 1, C)
        + torch.einsum("tij,tic->tjc", comb.double().reshape(T, N, N), x.double().reshape(T, N, C))
    ).reshape(T, nc)
    w = torch.zeros(nc, 32)
    raw = torch.randn(nc, NG) * nc**-0.5
    i = raw.view(torch.int32)
    sh = 14
    i = ((i + (1 << (sh - 1)) - 1 + ((i >> sh) & 1)) >> sh) << sh
    w[:, :NG] = i.view(torch.float32)
    d = lambda t, dt=ttnn.float32: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)

    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()

    for fid in (ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.LoFi):
        for grid in (None, ttnn.CoreCoord(1, 1)):
            cfg = ttnn.ComputeConfigDescriptor(math_fidelity=fid, fp32_dest_acc_en=True, math_approx_mode=False)
            xo, st = mhc_post(
                d(f, ttnn.bfloat16),
                d(x, ttnn.bfloat16),
                d(post),
                d(comb),
                stats_weight=d(w.reshape(1, 1, nc, 32)),
                compute_kernel_config=cfg,
                max_grid=grid,
            )
            xo = ttnn.to_torch(xo).double().reshape(T, nc)
            st = ttnn.to_torch(st).double().reshape(T, 32)
            print(
                "H",
                fid,
                grid,
                "proj_vs_xo",
                rel(st[:, :NG], (xo @ w.double())[:, :NG]),
                "proj_vs_mix",
                rel(st[:, :NG], (mix @ w.double())[:, :NG]),
                "ss_vs_mix",
                rel(st[:, NG], (mix**2).sum(1)),
                "xo_vs_mixbf16",
                (xo != mix.float().bfloat16().double()).sum().item(),
            )
finally:
    ttnn.close_device(device)
