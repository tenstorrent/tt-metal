import torch, ttnn
from ttnn.operations.mhc_post import mhc_post

N = 4
NG = 24
device = ttnn.open_device(device_id=0)


def trunc(t, bits):
    i = t.float().contiguous().view(torch.int32)
    return (i & ~((1 << (23 - bits)) - 1)).view(torch.float32).double()


try:
    torch.manual_seed(0)
    T, C = 32, 64
    f = torch.randn(1, 1, T, C).bfloat16()
    x = torch.randn(1, 1, T, N * C).bfloat16()
    post = (2 * torch.sigmoid(torch.randn(1, 1, T, N))).float()
    comb = torch.softmax(torch.randn(1, 1, T, N * N), -1).float()
    nc = N * C
    w = torch.zeros(nc, 32)
    raw = torch.randn(nc, NG) * nc**-0.5
    i = raw.view(torch.int32)
    sh = 14
    i = ((i + (1 << (sh - 1)) - 1 + ((i >> sh) & 1)) >> sh) << sh
    w[:, :NG] = i.view(torch.float32)
    d = lambda t, dt=ttnn.float32: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)

    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()

    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
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
    for wb in (6, 7, 8, 9):
        for xb in (5, 6, 7):
            print("H wbits", wb, "xbits", xb, rel(st, (trunc(xo, xb) @ trunc(w, wb))[:, :NG]))
    print("H exact", rel(st, (xo @ w.double())[:, :NG]))
    # ones test: X'=? use W = identity-like pattern
finally:
    ttnn.close_device(device)
