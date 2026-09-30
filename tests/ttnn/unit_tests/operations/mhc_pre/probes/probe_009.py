import torch, ttnn
from eval.golden_tests.mhc_pre.helpers import make_compute_config
from ttnn.operations.mhc_pre import mhc_pre

device = ttnn.open_device(device_id=0)


def bf(t):
    return t.to(torch.bfloat16).float()


try:
    for nC in (128, 1024, 4096):
        for mode in ("bf16vals", "fp32vals"):
            g = torch.Generator().manual_seed(3)
            x = torch.randn((64, nC), generator=g)
            w = torch.randn((nC, 24), generator=g) / nC**0.5
            if mode == "bf16vals":
                x, w = bf(x), bf(w)
            else:
                w = (w.view(torch.int32) & ~0x1FFF).view(torch.float32)
            b = torch.zeros((1, 24))
            dev = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            _, pd, _ = mhc_pre(
                dev(x), dev(w), dev(b), scale=(1.0, 1.0, 1.0), compute_kernel_config=make_compute_config()
            )
            p = ttnn.to_torch(pd).double().reshape(-1, 4)
            X = x.double()
            W = w.double()
            r = torch.rsqrt(X.square().mean(-1, keepdim=True) + 1e-6)
            z_ref = (X @ W)[:, 4:8] * r
            z_dev = torch.log((p / 2) / (1 - p / 2))
            rel = (z_dev - z_ref) / z_ref.abs().clamp(min=1e-3)
            d = z_dev - z_ref
            print(
                "PROBE",
                nC,
                mode,
                "abs rms %.3e mean %.3e" % (d.pow(2).mean().sqrt().item(), d.mean().item()),
                "median rel %.3e" % rel.abs().median().item(),
            )
finally:
    ttnn.close_device(device)
