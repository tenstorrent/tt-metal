import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as desc

dev = ttnn.open_device(device_id=0)


def measure(x, dtype=ttnn.bfloat16, iters=7):
    ti = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=dev)
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
    # Regime A: vary rows-per-core. 64 cores, Wt=8 (W=256). H in tiles -> rows/core.
    for Htiles in [64, 128, 256, 512, 1024]:
        H = Htiles * 32
        rpc = (Htiles + 63) // 64
        x = torch.randn(1, 1, H, 256)
        t = measure(x)
        print("RES H_tiles=%-5d rows/core=%-3d : %8.0f ns  (%.0f ns/row-per-core)" % (Htiles, rpc, t, t / rpc))
finally:
    ttnn.close_device(dev)
