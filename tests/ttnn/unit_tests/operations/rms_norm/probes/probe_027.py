import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as desc

dev = ttnn.open_device(device_id=0)


def measure(x, layout, dtype=ttnn.bfloat16, iters=7):
    ti = ttnn.from_torch(x, dtype=dtype, layout=layout, device=dev)
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


# Crossover-zone shapes: R5 routed all to B (adds cores); R6 keeps narrow ones in A.
shapes = [(1, 1, 32, 1024), (1, 1, 32, 2048), (1, 1, 32, 4096), (1, 1, 32, 8192)]
try:
    for lt, ltag in [(ttnn.TILE_LAYOUT, "TILE"), (ttnn.ROW_MAJOR_LAYOUT, "RM")]:
        for s in shapes:
            x = torch.randn(*s)
            desc._FORCE_REGIME = "B"
            old = measure(x, lt)  # R5: always B
            desc._FORCE_REGIME = None
            new = measure(x, lt)  # R6: heuristic
            sp = old / new if (old and new) else 0
            print(
                "RES %-4s Wt=%-4d old_B=%8.1f new=%8.1f speedup=%.2fx" % (ltag, s[-1] // 32, old or -1, new or -1, sp)
            )
finally:
    desc._FORCE_REGIME = None
    ttnn.close_device(dev)
