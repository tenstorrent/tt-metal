import torch, ttnn
from eval.golden_tests.mhc_pre.helpers import make_inputs, pytorch_mhc_pre, make_compute_config
from ttnn.operations.mhc_pre import mhc_pre

dev = ttnn.open_device(device_id=0)
for xd in (ttnn.float32, ttnn.bfloat16):
    for wd in (ttnn.float32, ttnn.bfloat16):
        xs = (1, 1, 32, 512)
        x, w, b, scale = make_inputs(xs, (xs[-1], 24), dtype=xd, weight_dtype=wd, seed=0)
        refs = pytorch_mhc_pre(x, w, b, scale=scale)
        d = lambda t, dt: ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
        outs = mhc_pre(d(x, xd), d(w, wd), d(b, ttnn.float32), scale=scale, compute_kernel_config=make_compute_config())
        for name, o, r in zip(("y", "post", "comb"), outs, refs):
            g = ttnn.to_torch(o).double().reshape(r.shape)
            r = r.double()
            rr = ((g - r).square().mean().sqrt() / r.std()).item()
            print("RES", xd, wd, name, f"{rr:.3e}", "g0", g.flatten()[:4].tolist(), "r0", r.flatten()[:4].tolist())
ttnn.close_device(dev)
