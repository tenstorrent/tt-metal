import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
_KEY = "DEVICE KERNEL DURATION [ns]"
def dev_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total = 0.0; found = False
    for programs in (per_chip or {}).values():
        for p in programs:
            res = getattr(p, "program_analyses_results", None) or {}
            e = res.get(_KEY)
            if e is None: continue
            total += float(e.duration); found = True
    return total if found else None
def measure(device, x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, iters=9):
    ti = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device)
    rms_norm(ti); dev_ns(device)
    best = None
    for _ in range(iters):
        rms_norm(ti); ns = dev_ns(device)
        if ns is not None: best = ns if best is None else min(best, ns)
    return best
device = ttnn.open_device(device_id=0)
try:
    for shp in [(1,1,32,8192),(1,1,32,16384),(1,1,32,32768),(1,1,64,12288)]:
        x = torch.randn(*shp); ns = measure(device, x)
        print(f"MEASURE shape={shp} OLD_combine device_ns={ns} ({None if ns is None else round(ns/1000,2)} us)")
finally:
    ttnn.close_device(device)
