import torch, ttnn
from eval.golden_tests.mhc_pre.helpers import make_compute_config
from ttnn.operations.mhc_pre import mhc_pre

device = ttnn.open_device(device_id=0)


def bf(t):
    return t.to(torch.bfloat16).float()


try:
    nC = 1024
    g = torch.Generator().manual_seed(3)
    x = bf(torch.randn((64, nC), generator=g))
    w = bf(torch.randn((nC, 24), generator=g) / nC**0.5)
    dev = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    for bval in (0.0, 0.5):
        b = torch.full((1, 24), bval)
        _, pd, _ = mhc_pre(dev(x), dev(w), dev(b), scale=(1.0, 1.0, 1.0), compute_kernel_config=make_compute_config())
        p = ttnn.to_torch(pd).double().reshape(-1, 4)
        X = x.double()
        W = w.double()
        r = torch.rsqrt(X.square().mean(-1, keepdim=True) + 1e-6)
        z_ref = (X @ W)[:, 4:8] * r + bval
        z_dev = torch.log((p / 2) / (1 - p / 2))
        ratio = (z_dev - bval) / (z_ref - bval)
        print("PROBE b", bval, "ratio-1 per row (first 6 rows):")
        for i in range(6):
            print(
                "PROBE", [f"{v:+.2e}" for v in (ratio[i] - 1).tolist()], "z", [f"{v:+.3f}" for v in z_ref[i].tolist()]
            )
        d = z_dev - z_ref
        print("PROBE abs err rows", [f"{v:+.2e}" for v in d[:3].flatten().tolist()])
finally:
    ttnn.close_device(device)
