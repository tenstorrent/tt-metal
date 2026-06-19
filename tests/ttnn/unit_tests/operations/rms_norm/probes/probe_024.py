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


try:
    print("RES RM crossover (1,1,32,W) bf16:")
    for W in [2048, 4096, 6144, 8192, 12288]:
        x = torch.randn(1, 1, 32, W)
        desc._FORCE_REGIME = "A"
        a = measure(x, ttnn.ROW_MAJOR_LAYOUT)
        desc._FORCE_REGIME = "B"
        b = measure(x, ttnn.ROW_MAJOR_LAYOUT)
        desc._FORCE_REGIME = None
        print(
            "RES Wt=%-4d W=%-6d A=%8.0f B=%8.0f winner=%s"
            % (W // 32, W, a or -1, b or -1, "A" if (a or 9e9) < (b or 9e9) else "B")
        )
finally:
    desc._FORCE_REGIME = None
    ttnn.close_device(dev)
