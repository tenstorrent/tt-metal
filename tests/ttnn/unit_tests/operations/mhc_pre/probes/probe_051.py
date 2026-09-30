import torch, ttnn, os, json
from eval.golden_tests.mhc_pre.helpers import make_inputs, pytorch_mhc_pre, make_compute_config
from ttnn.operations.mhc_pre import mhc_pre
import ttnn.operations.mhc_pre.mhc_pre_program_descriptor as pd

dev = ttnn.open_device(device_id=0)
xs = tuple(json.loads(os.environ.get("XS", "[1,1,1280,16384]")))
XD = getattr(ttnn, os.environ.get("XD", "float32"))
WD = getattr(ttnn, os.environ.get("WD", "bfloat16"))
CFGS = json.loads(os.environ.get("CFGS", '[{}, {"READER_NOC_FLIP_ROWS": 0}, {"W_SHARE_ON_READER": false}]'))
N = int(os.environ.get("N", 8))
d = lambda t, dt: ttnn.from_torch(
    t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
data = []
for seed in range(2):
    x, w, b, scale = make_inputs(xs, (xs[-1], 24), dtype=XD, weight_dtype=WD, seed=seed)
    data.append((d(x, XD), d(w, WD), d(b, ttnn.float32), scale, pytorch_mhc_pre(x, w, b, scale=scale)))
for knobs in CFGS:
    saved = {k: getattr(pd, k) for k in knobs}
    for k, v in knobs.items():
        setattr(pd, k, v)
    bad = 0
    for rep in range(N):
        tx, tw, tb, scale, refs = data[rep % 2]
        outs = mhc_pre(tx, tw, tb, scale=scale, compute_kernel_config=make_compute_config())
        for name, o, r in zip(("y", "post", "comb"), outs, refs):
            g = ttnn.to_torch(o).double().flatten()
            r = r.double().flatten()
            rr = ((g - r).square().mean().sqrt() / r.std()).item()
            if name != "y" and rr > 5e-4:
                bad += 1
                print("BAD", knobs, rep, name, rr)
                break
    print("RES", knobs, "bad", bad, "of", N)
    for k, v in saved.items():
        setattr(pd, k, v)
ttnn.close_device(dev)
