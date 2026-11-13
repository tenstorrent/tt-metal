"""greedy sampling stage deltas + alternatives; sampler (plain / full-vocab / topk path) traced latency"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "head_device.py")).read().split("for v in VARIANTS:")[0])
head = Hy4DeviceHead(md, mc, ccl, W, head_dtype="bf16", out_dtype="fp32", fidelity="HiFi4")
rows, cols = 1, 8
for T in TS:
    s = streams_tt(X[:T])
    lg = head.forward(s)
    st = {}

    def run(upto):
        mx = ttnn.max(lg, dim=-1, keepdim=True)
        if upto == 1:
            return mx
        rm = ttnn.to_layout(lg, ttnn.ROW_MAJOR_LAYOUT)
        if upto == 2:
            return rm
        idx = ttnn.argmax(rm, dim=-1, keepdim=True)
        if upto == 3:
            return idx
        idxf = ttnn.typecast(ttnn.to_layout(idx, ttnn.TILE_LAYOUT), ttnn.float32)
        if upto == 4:
            return idxf
        mxa = mc.allgather(mx, ccl, axis=1, dim=3)
        if upto == 5:
            return mxa
        ixa = mc.allgather(idxf, ccl, axis=1, dim=3)
        if upto == 6:
            return ixa
        return head.sample_global(lg)

    prev = 0
    for u in range(1, 7):
        us = timed_trace(lambda: run(u))
        res(event="gprof", T=T, upto=u, cum_us=round(us, 1), delta_us=round(us - prev, 1))
        prev = us
    # sampler
    smp = Hy4DeviceSampler(md, mc, ccl, T)
    prm = smp.alloc_params()
    for nm, tk, tp in [("plain", 0, 1.0), ("full_topp", 0, 0.95), ("topk20", 20, 1.0)]:
        invT, k, p, u, g = sampling_rows(0.9, tk, tp, torch.Generator().manual_seed(1), T)
        smp.set_params(invT, k, p, u, g * 0)
        body = lambda: smp.run(lg, smp.params, smp.fast)
        try:
            res(event="sampler", T=T, path=nm, plain=smp.plain, fast=smp.fast, us=round(timed_trace(body, 20), 1))
        except Exception as e:
            res(event="sampler", T=T, path=nm, err=str(e)[:300])
for T in TS:
    st = streams_tt(X[:T])
    res(event="chain", T=T, what="head+greedy", us=round(timed_trace(lambda: head.sample_global(head.forward(st))), 1))
    smp = Hy4DeviceSampler(md, mc, ccl, T)
    smp.alloc_params()
    invT, k, p, u, g = sampling_rows(0.9, 0, 1.0, torch.Generator().manual_seed(1), T)
    smp.set_params(invT, k, p, u, g * 0)
    res(
        event="chain",
        T=T,
        what="head+sampler_plain",
        plain=smp.plain,
        us=round(timed_trace(lambda: smp.run(head.forward(st), smp.params, smp.fast), 20), 1),
    )
    invT, k, p, u, g = sampling_rows(0.9, 20, 1.0, torch.Generator().manual_seed(1), T)
    smp.set_params(invT, k, p, u, g * 0)
    res(
        event="chain",
        T=T,
        what="head+sampler_topk20",
        fast=smp.fast,
        us=round(timed_trace(lambda: smp.run(head.forward(st), smp.params, smp.fast), 20), 1),
    )
ttnn.close_mesh_device(md)
