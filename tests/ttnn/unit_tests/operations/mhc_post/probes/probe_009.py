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
    nc = N * C
    f = torch.zeros(1, 1, T, C).bfloat16()
    x = rb(torch.randn(1, 1, T, nc), 3).bfloat16()
    post = torch.ones(1, 1, T, N)
    comb = torch.eye(N).reshape(1, 1, 1, N * N).expand(1, 1, T, N * N).contiguous().float()
    d = lambda t, dt=ttnn.float32: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)

    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()

    for fid in (ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4):
        cfg = ttnn.ComputeConfigDescriptor(math_fidelity=fid, fp32_dest_acc_en=True, math_approx_mode=False)
        for name, rows in (
            ("two_in_face", torch.tensor([3, 5])),
            ("face_dense16", torch.arange(16)),
            ("dense", torch.arange(nc)),
        ):
            w = torch.zeros(nc, 32)
            torch.manual_seed(1)
            for k in range(NG):
                w[rows, k] = rb(torch.randn(len(rows)), 3)
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
                name,
                "xo==x",
                torch.equal(xo, x.double().reshape(T, nc)),
                "rel %.3g maxabs %.3g" % (rel(st, ref), (st - ref).abs().max().item()),
            )
finally:
    ttnn.close_device(device)
