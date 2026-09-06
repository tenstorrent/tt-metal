import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")
import statistics, torch, ttnn

_K = "DEVICE KERNEL DURATION [ns]"


def ns(device):
    ttnn.ReadDeviceProfiler(device)
    tot = 0.0
    f = False
    for progs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for p in progs:
            e = (getattr(p, "program_analyses_results", None) or {}).get(_K)
            if e is not None:
                tot += float(e.duration)
                f = True
    return tot if f else None


def bench(device, fn, n=5):
    for _ in range(2):
        fn()
    ttnn.synchronize_device(device)
    ns(device)
    s = []
    for _ in range(n):
        o = fn()
        ttnn.synchronize_device(device)
        v = ns(device)
        if v:
            s.append(v)
        ttnn.deallocate(o)
    return statistics.median(s)


device = ttnn.open_device(device_id=0)
try:
    for shape in [(1, 1, 8192, 1024), (1, 1, 8192, 7168)]:
        t = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        x = ttnn.from_torch(
            t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        by = shape[2] * shape[3] * 2 * 2
        for name, fn in [
            ("exp", lambda: ttnn.exp(x)),
            ("mul_s", lambda: ttnn.multiply(x, 2.0)),
            ("clone", lambda: ttnn.clone(x)),
        ]:
            try:
                v = bench(device, fn)
                print(f"ROOF {shape} {name:6s} {v:10.0f} ns  {by/v:7.1f} GB/s")
            except Exception as e:
                print("ROOF", shape, name, "ERR", e)
        ttnn.deallocate(x)
finally:
    ttnn.close_device(device)
