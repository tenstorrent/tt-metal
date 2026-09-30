import torch, ttnn
from eval.golden_tests.mhc_pre.helpers import make_inputs, pytorch_mhc_pre, make_compute_config
from ttnn.operations.mhc_pre import mhc_pre


def trunc(t, drop):
    i = t.contiguous().view(torch.int32)
    return (i & ~((1 << drop) - 1)).view(torch.float32)


def relrms(a, b):
    a = a.double().flatten()
    b = b.double().flatten()
    return ((a - b).square().mean().sqrt() / b.std()).item()


device = ttnn.open_device(device_id=0)
try:
    shape = (1, 1, 256, 24576)
    x, w, b, s = make_inputs(shape, (shape[-1], 24), dtype=ttnn.bfloat16, weight_dtype=ttnn.float32, seed=0)
    dev = lambda t, d: ttnn.from_torch(t, dtype=d, layout=ttnn.TILE_LAYOUT, device=device)
    for drop, label in [(13, "tf32 W (golden)"), (14, "9-bit W"), (16, "bf16-valued W")]:
        wt = trunc(w, drop)
        y, p, c = pytorch_mhc_pre(x, wt, b, scale=s)
        yd, pd, cd = mhc_pre(
            dev(x, ttnn.bfloat16),
            dev(wt, ttnn.float32),
            dev(b, ttnn.float32),
            scale=s,
            compute_kernel_config=make_compute_config(),
        )
        print(
            "PROBE", label, "post", f"{relrms(ttnn.to_torch(pd),p):.2e}", "comb", f"{relrms(ttnn.to_torch(cd),c):.2e}"
        )
finally:
    ttnn.close_device(device)
