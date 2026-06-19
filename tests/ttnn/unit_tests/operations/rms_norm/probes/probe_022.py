import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as desc

dev = ttnn.open_device(device_id=0)


def measure(x, gamma=None, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, iters=7):
    ti = ttnn.from_torch(x, dtype=dtype, layout=layout, device=dev)
    g = ttnn.from_torch(gamma, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev) if gamma is not None else None
    rms_norm(ti, gamma=g)
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    s = []
    for _ in range(iters):
        rms_norm(ti, gamma=g)
        ttnn.synchronize_device(dev)
        ttnn.ReadDeviceProfiler(dev)
        pc = ttnn.get_latest_programs_perf_data()
        tot = 0.0
        f = False
        for progs in (pc or {}).values():
            for p in progs:
                e = (getattr(p, "program_analyses_results", None) or {}).get("DEVICE KERNEL DURATION [ns]")
                if e is not None:
                    tot += float(e.duration)
                    f = True
        if f:
            s.append(tot)
    return min(s) if s else None


try:
    print("Wt  | W      | A-forced(ns) | B-forced(ns) | winner | speedup")
    for W in [1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384]:
        Wt = W // 32
        x = torch.randn(1, 1, 32, W)
        desc._FORCE_REGIME = "A"
        a = measure(x)
        desc._FORCE_REGIME = "B"
        b = measure(x)
        desc._FORCE_REGIME = None
        win = "A" if (a or 9e9) < (b or 9e9) else "B"
        sp = (b / a) if (a and b) else 0
        print("RES %-3d | %-6d | %12.0f | %12.0f | %s | %.2f" % (Wt, W, a or -1, b or -1, win, sp))
finally:
    desc._FORCE_REGIME = None
    ttnn.close_device(dev)
