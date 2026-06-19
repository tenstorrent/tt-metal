import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
try:
    x = torch.randn(1, 1, 2048, 256)
    ti = ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    rms_norm(ti)
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    data = ttnn.get_latest_programs_perf_data()
    n = 0
    for progs in (data or {}).values():
        for p in progs:
            r = getattr(p, "program_analyses_results", None) or {}
            if "DEVICE KERNEL DURATION [ns]" in r:
                n += 1
    print("PROFILER_AVAILABLE" if n > 0 else "NO_PROFILER_DATA", "programs_with_duration=", n)
finally:
    ttnn.close_device(dev)
