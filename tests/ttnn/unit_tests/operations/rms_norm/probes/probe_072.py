import torch, ttnn
from ttnn.operations.rms_norm import rms_norm


def tot(dev):
    ttnn.ReadDeviceProfiler(dev)
    per = ttnn.get_latest_programs_perf_data()
    t = 0.0
    for pr in (per or {}).values():
        for p in pr:
            r = getattr(p, "program_analyses_results", None) or {}
            e = r.get("DEVICE KERNEL DURATION [ns]")
            t += float(e.duration) if e else 0
    return t


dev = ttnn.open_device(device_id=0)
try:
    print("PERF_BEGIN")
    for shp in [(1, 1, 512, 8192), (1, 1, 512, 4096), (1, 1, 1024, 8192)]:
        x = torch.randn(shp, dtype=torch.float32)
        ti = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        rms_norm(ti)
        ttnn.ReadDeviceProfiler(dev)
        best = min(((rms_norm(ti), tot(dev))[1] for _ in range(3)))
        print(f"  {str(shp):18s} total_device_kernel_ns={best:.0f}")
    print("PERF_END")
finally:
    ttnn.close_device(dev)
