import torch, ttnn
from eval.golden_tests.mhc_pre.helpers import make_compute_config
from ttnn.operations.mhc_pre import mhc_pre

device = ttnn.open_device(device_id=0)


def keep(t, bits):
    drop = 24 - bits
    return (t.view(torch.int32) & ~((1 << drop) - 1)).view(torch.float32)


try:
    nC = 1024
    g = torch.Generator().manual_seed(3)
    x = keep(torch.randn((64, nC), generator=g), 3)
    b = torch.zeros((1, 24))
    for label, nnz in (("onehot", 1), ("2hot", 2), ("32hot same tile", 32), ("32hot spread", -32)):
        w = torch.zeros((nC, 24))
        for k in range(24):
            if nnz > 0:
                for j in range(nnz):
                    w[(k * 37 + j) % nC, k] = 0.25 * (1 + j % 3)
            else:
                for j in range(32):
                    w[(k * 37 + j * 32) % nC, k] = 0.25 * (1 + j % 3)
        dev = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        _, pd, _ = mhc_pre(dev(x), dev(w), dev(b), scale=(1.0, 1.0, 1.0), compute_kernel_config=make_compute_config())
        p = ttnn.to_torch(pd).double().reshape(-1, 4)
        X = x.double()
        W = w.double()
        r = torch.rsqrt(X.square().mean(-1, keepdim=True) + 1e-6)
        mix = (X @ W)[:, 4:8]
        z_ref = mix * r
        z_dev = torch.log((p / 2) / (1 - p / 2))
        mix_dev = z_dev / r
        d = mix_dev - mix
        print(
            "PROBE",
            label,
            "mix abs err rms %.3e max %.3e" % (d.pow(2).mean().sqrt().item(), d.abs().max().item()),
            "sample",
            [f"{a:.6f}/{b_:.6f}" for a, b_ in zip(mix_dev[0].tolist(), mix[0].tolist())],
        )
finally:
    ttnn.close_device(device)
