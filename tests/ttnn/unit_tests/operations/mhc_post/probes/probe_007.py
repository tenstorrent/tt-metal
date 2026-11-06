import torch, ttnn
from ttnn.operations.mhc_post import mhc_post

N = 4
NG = 24
device = ttnn.open_device(device_id=0)


def rb(t, bits):
    i = t.float().contiguous().view(torch.int32)
    sh = 23 - bits
    return (((i + (1 << (sh - 1)) - 1 + ((i >> sh) & 1)) >> sh) << sh).view(torch.float32)


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
    d = lambda t, dt=ttnn.float32: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)

    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()

    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    for name, rows in (
        ("tile0_dense32", torch.arange(32)),
        ("face_dense16", torch.arange(16)),
        ("one_per_tile", torch.arange(0, nc, 32)),
        ("dense", torch.arange(nc)),
    ):
        w = torch.zeros(nc, 32)
        torch.manual_seed(1)
        for k in range(NG):
            w[rows, k] = rb(torch.randn(len(rows)), 4)
        xo, st = mhc_post(
            d(f, ttnn.bfloat16),
            d(x, ttnn.bfloat16),
            d(post),
            d(comb),
            stats_weight=d(w.reshape(1, 1, nc, 32)),
            compute_kernel_config=cfg,
        )
        xo = ttnn.to_torch(xo).double().reshape(T, nc)
        st = ttnn.to_torch(st).double().reshape(T, 32)[:, :NG]
        print(
            "H",
            name,
            "vs_xo %.3g vs_mix %.3g"
            % (rel(st, (xo @ w.double())[:, :NG]), rel(st, (mix.float().double() @ w.double())[:, :NG])),
        )
finally:
    ttnn.close_device(device)
