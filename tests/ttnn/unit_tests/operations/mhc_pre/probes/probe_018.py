import torch, ttnn
from eval.golden_tests.mhc_pre.helpers import make_inputs, pytorch_mhc_pre, make_compute_config
from ttnn.operations.mhc_pre import mhc_pre

device = ttnn.open_device(device_id=0)
try:
    x, w, b, scale = make_inputs(
        (1, 1, 64, 4096), (4096, 24), dtype=ttnn.float32, weight_dtype=ttnn.float32, seed=42, logit_scale=30.0
    )
    dev = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    _, pd, cd = mhc_pre(dev(x), dev(w), dev(b), scale=scale, compute_kernel_config=make_compute_config())
    p = ttnn.to_torch(pd).double().reshape(-1, 4)
    c = ttnn.to_torch(cd).double().reshape(-1, 4, 4)
    X = x.double().reshape(-1, 4096)
    W = w.double()
    r = torch.rsqrt(X.square().mean(-1, keepdim=True) + 1e-6)
    mix = X @ W
    z_ref = mix[:, 4:8] * r + b.double().reshape(-1)[4:8]
    z_dev = torch.log((p / 2) / (1 - p / 2))
    d = z_dev - z_ref
    rel = d / (mix[:, 4:8] * r)
    print(
        "PROBE z err max",
        d.abs().max().item(),
        "rms",
        d.pow(2).mean().sqrt().item(),
        "mean rel",
        rel.mean().item(),
        "rms rel",
        rel.pow(2).mean().sqrt().item(),
    )
    _, _, cref = pytorch_mhc_pre(x, w, b, scale=scale)
    print(
        "PROBE rowerr dev",
        (c.sum(-1) - 1).abs().max().item(),
        "ref",
        (cref.double().reshape(-1, 4, 4).sum(-1) - 1).abs().max().item(),
    )
finally:
    ttnn.close_device(device)
