import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as desc

dev = ttnn.open_device(device_id=0)
try:
    KEY = "DEVICE KERNEL DURATION [ns]"

    def read_ns():
        ttnn.ReadDeviceProfiler(dev)
        per = ttnn.get_latest_programs_perf_data()
        tot = 0.0
        f = False
        for progs in (per or {}).values():
            for p in progs:
                r = getattr(p, "program_analyses_results", None) or {}
                e = r.get(KEY)
                if e is None:
                    continue
                tot += float(e.duration)
                f = True
        return tot if f else None

    def meas(x, iters=7):
        ti = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        rms_norm(ti)
        ttnn.synchronize_device(dev)
        ttnn.ReadDeviceProfiler(dev)
        s = []
        for _ in range(iters):
            rms_norm(ti)
            ttnn.synchronize_device(dev)
            ns = read_ns()
            if ns:
                s.append(ns)
        return min(s) if s else None

    x1 = torch.randn(1, 1, 32, 8192)
    x2 = torch.randn(1, 1, 64, 8192)
    t1 = meas(x1)
    t2 = meas(x2)
    print(f"MEAS nrg1_K16={t1/1000:.2f}us nrg2_K16={t2/1000:.2f}us RATIO {t2/t1:.2f}")
    for K in (16, 32, 64):
        desc._FORCE_K = K
        t = meas(x1)
        desc._FORCE_K = None
        print(f"MEAS forceK={K}: {t/1000:.2f}us")
finally:
    ttnn.close_device(dev)
