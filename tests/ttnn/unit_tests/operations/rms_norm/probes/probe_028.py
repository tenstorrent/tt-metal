import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as desc

dev = ttnn.open_device(device_id=0)


def measure(x, layout, iters=7):
    ti = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=dev)
    rms_norm(ti)
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    s = []
    for _ in range(iters):
        rms_norm(ti)
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
    for shape in [(1, 1, 32, 8192), (1, 1, 32, 16384)]:
        x = torch.randn(*shape)
        print("RES shape=%s Wt=%d:" % (shape, shape[-1] // 32))
        for K in [8, 16, 32, 64]:
            desc._FORCE_K = K
            try:
                t = measure(x, ttnn.TILE_LAYOUT)
            except Exception as e:
                t = None
            desc._FORCE_K = None
            print("RES   K=%-3d -> %s ns" % (K, ("%.0f" % t) if t else "infeasible/err"))
finally:
    desc._FORCE_K = None
    ttnn.close_device(dev)
