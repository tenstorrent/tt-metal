import torch, ttnn
from eval.golden_tests.mhc_pre.helpers import make_inputs, pytorch_mhc_pre, make_compute_config
from ttnn.operations.mhc_pre import mhc_pre

dev = ttnn.open_device(device_id=0)
bad = 0
for seed in range(40):
    xs = (1, 1, 256, 24576)
    x, w, b, scale = make_inputs(xs, (xs[-1], 24), dtype=ttnn.bfloat16, weight_dtype=ttnn.float32, seed=seed)
    refs = pytorch_mhc_pre(x, w, b, scale=scale)
    d = lambda t, dt: ttnn.from_torch(
        t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    outs = mhc_pre(
        d(x, ttnn.bfloat16),
        d(w, ttnn.float32),
        d(b, ttnn.float32),
        scale=scale,
        compute_kernel_config=make_compute_config(),
    )
    for name, o, r in zip(("y", "post", "comb"), outs, refs):
        g = ttnn.to_torch(o).double().flatten()
        r = r.double().flatten()
        rr = ((g - r).square().mean().sqrt() / r.std()).item()
        if name != "y" and rr > 5e-4:
            bad += 1
            print("BAD", seed, name, rr)
print("STRESS bad =", bad)
ttnn.close_device(dev)
