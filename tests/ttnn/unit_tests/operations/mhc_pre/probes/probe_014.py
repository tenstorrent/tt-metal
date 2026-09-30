import torch, ttnn
from eval.golden_tests.mhc_pre.helpers import make_compute_config
from ttnn.operations.mhc_pre import mhc_pre

device = ttnn.open_device(device_id=0)
try:
    nC = 128
    b = torch.zeros((1, 24))
    g = torch.Generator().manual_seed(5)
    for xbits, wbits in ((1, 2), (1, 4), (1, 6), (1, 7), (1, 8), (2, 2), (3, 3), (4, 4)):
        # x: 32 rows of integers in +-[0, 2^xbits-1] (nonzero), W: ints in +-[0, 2^wbits-1]; K = 128 (4 tiles) but only tile 0 nonzero
        xi = torch.randint(1, 2**xbits, (32, nC), generator=g).float() * (
            torch.randint(0, 2, (32, nC), generator=g) * 2 - 1
        )
        wi = torch.randint(0, 2**wbits, (nC, 24), generator=g).float() * (
            torch.randint(0, 2, (nC, 24), generator=g) * 2 - 1
        )
        wi[32:] = 0
        mix_int = xi.double() @ wi.double()
        sc = 1.0 / (2**wbits * 2**xbits * 4)
        w = wi * sc
        dev = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        _, pd, _ = mhc_pre(dev(xi), dev(w), dev(b), scale=(1.0, 1.0, 1.0), compute_kernel_config=make_compute_config())
        p = ttnn.to_torch(pd).double().reshape(-1, 4)
        X = xi.double()
        r = torch.rsqrt(X.square().mean(-1, keepdim=True) + 1e-6)
        z_dev = torch.log((p / 2) / (1 - p / 2))
        mix_dev_int = z_dev / r / sc
        d = mix_dev_int - mix_int[:, 4:8]
        print(
            "PROBE xbits",
            xbits,
            "wbits",
            wbits,
            "max|sum| %.0f" % mix_int[:, 4:8].abs().max().item(),
            "max int err %.3f rms %.3f" % (d.abs().max().item(), d.pow(2).mean().sqrt().item()),
        )
finally:
    ttnn.close_device(device)
