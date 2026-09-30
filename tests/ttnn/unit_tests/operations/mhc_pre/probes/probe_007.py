import torch, ttnn
from eval.golden_tests.mhc_pre.helpers import make_compute_config
from ttnn.operations.mhc_pre import mhc_pre

device = ttnn.open_device(device_id=0)


def trunc(t, drop):
    i = t.contiguous().view(torch.int32)
    return (i & ~((1 << drop) - 1)).view(torch.float32)


try:
    g = torch.Generator().manual_seed(1)
    nC = 128
    x = (torch.randint(0, 2, (32, nC), generator=g) * 2 - 1).float()
    b = torch.randn((1, 24), generator=g)
    w_rand = trunc(torch.randn((nC, 24), generator=g) / nC**0.5, 16)
    w_hot = torch.zeros((nC, 24))
    for k in range(24):
        w_hot[k * 5 % nC, k] = 0.5 * (1 + (k % 3))
    dev = lambda t, d: ttnn.from_torch(t, dtype=d, layout=ttnn.TILE_LAYOUT, device=device)
    for label, w in [("rand W", w_rand), ("one-hot W", w_hot)]:
        for eps_n in (1e-6, 0.0):
            _, pd, _ = mhc_pre(
                dev(x, ttnn.bfloat16),
                dev(w, ttnn.float32),
                dev(b, ttnn.float32),
                scale=(1.0, 1.0, 1.0),
                norm_eps=max(eps_n, 1e-30),
                compute_kernel_config=make_compute_config(),
            )
            p = ttnn.to_torch(pd).double().reshape(-1, 4)
            xd = x.double()
            r = torch.rsqrt(xd.square().mean(-1, keepdim=True) + max(eps_n, 1e-30))
            z_ref = (xd @ w.double())[:, 4:8] * r + b.double().reshape(-1)[4:8]
            p_ref = 2 * torch.sigmoid(z_ref)
            z_dev = torch.log((p / 2) / (1 - p / 2))
            print(
                "PROBE",
                label,
                "norm_eps",
                eps_n,
                "max|z_dev-z_ref|",
                f"{(z_dev-z_ref).abs().max().item():.3e}",
                "max|post rel err|",
                f"{((p-p_ref)/p_ref).abs().max().item():.3e}",
            )
finally:
    ttnn.close_device(device)
