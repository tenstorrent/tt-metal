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
    d = lambda t, dt=ttnn.float32: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)

    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()

    for fid in (ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi3):
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=fid, fp32_dest_acc_en=True, math_approx_mode=False)
        for wbits in (4, 7, 9):
            for nnz, same_tile in ((2, True), (2, False), (256, None)):
                w = torch.zeros(nc, 32)
                torch.manual_seed(1)
                for k in range(NG):
                    if nnz == 2 and same_tile:
                        rows = torch.tensor([3, 17])
                    elif nnz == 2:
                        rows = torch.tensor([3, 64 + 5])
                    else:
                        rows = torch.arange(nc)
                    w[rows, k] = rb(torch.randn(len(rows)), wbits)
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
                ref = (xo @ w.double())[:, :NG]
                print(
                    "H",
                    fid,
                    "wbits",
                    wbits,
                    "nnz",
                    nnz,
                    "same_tile",
                    same_tile,
                    "rel %.3g maxabs %.3g" % (rel(st, ref), (st - ref).abs().max().item()),
                )
finally:
    ttnn.close_device(device)
