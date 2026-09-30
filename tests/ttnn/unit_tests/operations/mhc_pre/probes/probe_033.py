import torch, ttnn
from eval.golden_tests.mhc_pre.helpers import make_inputs, pytorch_mhc_pre, make_compute_config
from ttnn.operations.mhc_pre import mhc_pre
import ttnn.operations.mhc_pre.mhc_pre_program_descriptor as pd

dev = ttnn.open_device(device_id=0)
xs = (1, 1, 1280, 16384)
x, w, b, scale = make_inputs(xs, (xs[-1], 24), dtype=ttnn.float32, weight_dtype=ttnn.bfloat16, seed=0)
refs = pytorch_mhc_pre(x, w, b, scale=scale)
d = lambda t, dt: ttnn.from_torch(
    t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
for knobs in [
    dict(),
    dict(READER_NOC_FLIP_ROWS=0),
    dict(W_SHARE_ON_READER=False),
    dict(READER_NOC_FLIP_ROWS=0, W_SHARE_ON_READER=False),
    dict(),
]:
    saved = {k: getattr(pd, k) for k in knobs}
    for k, v in knobs.items():
        setattr(pd, k, v)
    for rep in range(2):
        outs = mhc_pre(
            d(x, ttnn.float32),
            d(w, ttnn.bfloat16),
            d(b, ttnn.float32),
            scale=scale,
            compute_kernel_config=make_compute_config(),
        )
        res = []
        for name, o, r in zip(("y", "post", "comb"), outs, refs):
            g = ttnn.to_torch(o).double().flatten()
            r = r.double().flatten()
            res.append("%s %.2e" % (name, ((g - r).square().mean().sqrt() / r.std()).item()))
        print("RES", knobs, rep, res)
    for k, v in saved.items():
        setattr(pd, k, v)
ttnn.close_device(dev)
