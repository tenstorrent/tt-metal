import torch, ttnn
from ttnn.operations.mhc_post import mhc_post

N = 4
NG = 24
device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    T, C = 32, 256
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
    st = ttnn.to_torch(st).double().reshape(T, 32)

    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()

    ref = xo @ w.double()
    print("H exact", rel(st[:, :NG], ref[:, :NG]))
    wb = w.bfloat16().double()
    print("H w_bf16_rne", rel(st[:, :NG], (xo @ wb)[:, :NG]))
    wt = (w.view(torch.int32) & ~0xFFFF).view(torch.float32).double()
    print("H w_bf16_trunc", rel(st[:, :NG], (xo @ wt)[:, :NG]))
    w7 = (w.view(torch.int32) & ~((1 << (23 - 6)) - 1)).view(torch.float32).double()
    print("H w_6bit", rel(st[:, :NG], (xo @ w7)[:, :NG]))
    # per-column / per-row structure
    err = (st[:, :NG] - ref[:, :NG]).abs()
    print("H err row max", err.max(1).values[:8].tolist())
    print("H err col max", err.max(0).values[:8].tolist())
    print("H ref sample", ref[0, :4].tolist(), st[0, :4].tolist())
    mix = None
    print("ss rel", rel(st[:, NG], (xo**2).sum(1)), st[0, NG].item(), (xo**2).sum(1)[0].item())
finally:
    ttnn.close_device(device)
