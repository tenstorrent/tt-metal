import torch, ttnn, os
from ttnn.operations.rms_norm import rms_norm

print("ENV profiler:", os.environ.get("TT_METAL_DEVICE_PROFILER"))
dev = ttnn.open_device(device_id=0)
try:

    def measure(x, gamma=None, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, iters=5):
        ti = ttnn.from_torch(x, dtype=dtype, layout=layout, device=dev)
        g = None
        if gamma is not None:
            g = ttnn.from_torch(gamma, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        # warmup
        out = rms_norm(ti, gamma=g)
        ttnn.synchronize_device(dev)
        ttnn.ReadDeviceProfiler(dev)  # flush
        samples = []
        for _ in range(iters):
            out = rms_norm(ti, gamma=g)
            ttnn.synchronize_device(dev)
            ttnn.ReadDeviceProfiler(dev)
            per_chip = ttnn.get_latest_programs_perf_data()
            tot = 0.0
            found = False
            for programs in (per_chip or {}).values():
                for p in programs:
                    res = getattr(p, "program_analyses_results", None) or {}
                    e = res.get("DEVICE KERNEL DURATION [ns]")
                    if e is not None:
                        tot += float(e.duration)
                        found = True
            if found:
                samples.append(tot)
        return min(samples) if samples else None

    # Regime A many-row, narrow W
    x = torch.randn(1, 1, 2048, 256)
    ns = measure(x)
    print("RegimeA (1,1,2048,256) bf16 no_gamma: %.1f ns" % ns)
    # Regime B wide W
    x2 = torch.randn(1, 1, 32, 8192)
    ns2 = measure(x2)
    print("RegimeB (1,1,32,8192) bf16 no_gamma: %.1f ns" % ns2)
finally:
    ttnn.close_device(dev)
