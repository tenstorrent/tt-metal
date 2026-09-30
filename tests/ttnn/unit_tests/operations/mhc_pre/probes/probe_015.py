import torch, ttnn
from eval.golden_tests.mhc_pre.helpers import make_compute_config
from ttnn.operations.mhc_pre import mhc_pre

device = ttnn.open_device(device_id=0)
try:
    nC = 128
    b = torch.zeros((1, 24))
    g = torch.Generator().manual_seed(5)
    for xmax, wmax, dense in ((16, 32, True), (16, 64, True), (16, 64, False), (32, 32, True), (32, 64, True)):
        xi = torch.randint(-xmax, xmax + 1, (32, nC), generator=g).float()
        wi = torch.randint(-wmax, wmax + 1, (nC, 24), generator=g).float()
        if not dense:  # worst case: all products at the max, same sign
            xi = torch.full((32, nC), float(xmax))
            wi = torch.full((nC, 24), float(wmax))
            wi[::2] = 1.0
        xi[:, 32:] = 0
        mix_int = xi.double() @ wi.double()
        sc = 1.0 / (xmax * wmax * 8)
        w = wi * sc
        dev = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        _, pd, _ = mhc_pre(dev(xi), dev(w), dev(b), scale=(1.0, 1.0, 1.0), compute_kernel_config=make_compute_config())
        p = ttnn.to_torch(pd).double().reshape(-1, 4)
        X = xi.double()
        r = torch.rsqrt(X.square().mean(-1, keepdim=True) + 1e-6)
        z_dev = torch.log((p / 2) / (1 - p / 2))
        z_ref = mix_int[:, 4:8] * sc * r
        d = z_dev - z_ref
        print(
            "PROBE x<=",
            xmax,
            "w<=",
            wmax,
            "dense" if dense else "worst",
            "max|z| %.2f" % z_ref.abs().max().item(),
            "z err max %.2e rms %.2e" % (d.abs().max().item(), d.pow(2).mean().sqrt().item()),
        )
finally:
    ttnn.close_device(device)
