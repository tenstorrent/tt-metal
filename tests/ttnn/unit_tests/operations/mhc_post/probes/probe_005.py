import torch, ttnn
from ttnn.operations.mhc_post import mhc_post

N = 4
NG = 24
device = ttnn.open_device(device_id=0)


def r9(t):
    i = t.float().contiguous().view(torch.int32)
    sh = 14
    return (((i + (1 << (sh - 1)) - 1 + ((i >> sh) & 1)) >> sh) << sh).view(torch.float32)


try:
    torch.manual_seed(0)
    T, C = 32, 64
    f = torch.randn(1, 1, T, C).bfloat16()
    x = torch.randn(1, 1, T, N * C).bfloat16()
    post = (2 * torch.sigmoid(torch.randn(1, 1, T, N))).float()
    comb = torch.softmax(torch.randn(1, 1, T, N * N), -1).float()
    nc = N * C
    d = lambda t, dt=ttnn.float32: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)

    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()

    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )

    def run(w):
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
        return xo, st

    for nnz in (1, 2, 8, 32, 256):
        w = torch.zeros(nc, 32)
        for k in range(NG):
            rows = torch.randperm(nc)[:nnz]
            w[rows, k] = r9(torch.randn(nnz))
        xo, st = run(w)
        ref = (xo @ w.double())[:, :NG]
        print("H nnz", nnz, rel(st, ref), "maxabs", (st - ref).abs().max().item())
    # constant one row per column
    w = torch.zeros(nc, 32)
    w[torch.arange(NG) * 7, torch.arange(NG)] = 1.0
    xo, st = run(w)
    ref = (xo @ w.double())[:, :NG]
    print("H onehot", rel(st, ref), (st - ref).abs().max().item(), st[0, :3].tolist(), ref[0, :3].tolist())
finally:
    ttnn.close_device(device)
