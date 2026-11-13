"""plain sampler stage deltas"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "head_device.py")).read().split("for v in VARIANTS:")[0])
head = Hy4DeviceHead(md, mc, ccl, W, head_dtype="bf16", out_dtype="fp32", fidelity="HiFi4")
for T in TS:
    lg = head.forward(streams_tt(X[:T]))
    smp = Hy4DeviceSampler(md, mc, ccl, T)
    smp.alloc_params()
    invT, k, p, u, g = sampling_rows(0.9, 0, 1.0, torch.Generator().manual_seed(1), T)
    smp.set_params(invT, k, p, u, g * 0)
    prev = 0
    for stop in ["greedy_part", "scale_exp", "prefix", None]:
        smp.stop = stop
        us = timed_trace(lambda: smp.run(lg, smp.params, smp.fast), 20)
        res(event="sprof", T=T, stop=stop, cum_us=round(us, 1), delta_us=round(us - prev, 1))
        prev = us
ttnn.close_mesh_device(md)
