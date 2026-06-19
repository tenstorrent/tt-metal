import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

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


shapes = [(1, 1, 32, 2048), (1, 1, 32, 4096), (1, 1, 32, 8192), (1, 1, 32, 16384), (1, 1, 2048, 256), (1024, 1024)]
try:
    for s in shapes:
        x = torch.randn(*s)
        print(
            "RES R6 %-16s TILE=%8.0f RM=%8.0f"
            % (str(s), measure(x, ttnn.TILE_LAYOUT) or -1, measure(x, ttnn.ROW_MAJOR_LAYOUT) or -1)
        )
finally:
    ttnn.close_device(dev)
