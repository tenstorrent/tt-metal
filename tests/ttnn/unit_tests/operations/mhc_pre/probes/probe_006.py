import torch, ttnn
from eval.golden_tests.mhc_pre.helpers import make_inputs, make_compute_config
from ttnn.operations.mhc_pre import mhc_pre


def trunc(t, drop):
    i = t.contiguous().view(torch.int32)
    return (i & ~((1 << drop) - 1)).view(torch.float32)


device = ttnn.open_device(device_id=0)
try:
    for shape in [(1, 1, 256, 24576), (1, 1, 64, 4 * 256), (32, 128)]:
        x, w, b, s = make_inputs(shape, (shape[-1], 24), dtype=ttnn.bfloat16, weight_dtype=ttnn.float32, seed=0)
        w = trunc(w, 16)  # bf16-valued: exact in the FPU
        dev = lambda t, d: ttnn.from_torch(t, dtype=d, layout=ttnn.TILE_LAYOUT, device=device)
        _, pd, _ = mhc_pre(
            dev(x, ttnn.bfloat16),
            dev(w, ttnn.float32),
            dev(b, ttnn.float32),
            scale=(1.0, 1.0, 1.0),
            compute_kernel_config=make_compute_config(),
        )
        p = ttnn.to_torch(pd).double().reshape(-1, 4)
        xd = x.double().reshape(-1, shape[-1])
        wd = w.double()
        bd = b.double().reshape(-1)
        ssq = xd.square().sum(-1, keepdim=True)
        raw = xd @ wd  # exact mix sums
        r = torch.rsqrt(ssq / shape[-1] + 1e-6)
        z_dev = torch.log((p / 2) / (1 - p / 2)) - bd[4:8]  # = mix*r on device
        ratio = z_dev / (raw[:, 4:8] * r)
        m = (raw[:, 4:8] * r).abs() > 0.3
        rr = torch.where(m, ratio, torch.nan)
        per_tok_med = rr.nanmedian(dim=1).values
        within = (rr - per_tok_med[:, None]).abs()
        print(
            "PROBE",
            shape,
            "median |ratio-1| =",
            f"{(rr-1).abs().nanmedian().item():.2e}",
            "| per-token common offset median",
            f"{(per_tok_med-1).abs().nanmedian().item():.2e}",
            "| within-token spread median",
            f"{within.nanmedian().item():.2e}",
        )
        # sumsq sensitivity: implied r error if common offset is from r
finally:
    ttnn.close_device(device)
