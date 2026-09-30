import torch, ttnn
from eval.golden_tests.mhc_pre.helpers import make_compute_config
from ttnn.operations.mhc_pre import mhc_pre

device = ttnn.open_device(device_id=0)


def keep(t, bits):  # keep `bits` significant bits (incl hidden), truncating
    drop = 24 - bits
    return (t.view(torch.int32) & ~((1 << drop) - 1)).view(torch.float32)


try:
    nC = 1024
    for xb, wb in ((3, 3), (5, 5), (6, 6), (8, 3), (8, 8), (4, 8)):
        g = torch.Generator().manual_seed(3)
        x = keep(torch.randn((64, nC), generator=g), xb)
        w = keep(torch.randn((nC, 24), generator=g) / nC**0.5, wb)
        b = torch.zeros((1, 24))
        dev = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        _, pd, _ = mhc_pre(dev(x), dev(w), dev(b), scale=(1.0, 1.0, 1.0), compute_kernel_config=make_compute_config())
        p = ttnn.to_torch(pd).double().reshape(-1, 4)
        X = x.double()
        W = w.double()
        r = torch.rsqrt(X.square().mean(-1, keepdim=True) + 1e-6)
        z_ref = (X @ W)[:, 4:8] * r
        z_dev = torch.log((p / 2) / (1 - p / 2))
        d = z_dev - z_ref
        print(
            "PROBE xbits", xb, "wbits", wb, "abs rms %.3e mean %.3e" % (d.pow(2).mean().sqrt().item(), d.mean().item())
        )
finally:
    ttnn.close_device(device)
