import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd


def read_ns(dev):
    ttnn.ReadDeviceProfiler(dev)
    data = ttnn.get_latest_programs_perf_data()
    tot = 0.0
    found = False
    for progs in (data or {}).values():
        for p in progs:
            r = getattr(p, "program_analyses_results", None) or {}
            e = r.get("DEVICE KERNEL DURATION [ns]")
            if e is not None:
                tot += float(e.duration)
                found = True
    return tot if found else None


def measure(dev, x, iters=9):
    ti = ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    rms_norm(ti)
    ttnn.synchronize_device(dev)
    ttnn.ReadDeviceProfiler(dev)
    s = []
    for _ in range(iters):
        rms_norm(ti)
        ttnn.synchronize_device(dev)
        ns = read_ns(dev)
        if ns is not None:
            s.append(ns)
    return min(s) if s else None


dev = ttnn.open_device(device_id=0)
try:
    shapes = [(1, 1, 4096, 256), (1, 1, 8192, 256), (1, 1, 16384, 256), (1, 1, 8192, 512)]
    print("RESULTS_START")
    print(f"{'shape':22}{'bh':>4}{'bh1_us':>10}{'bhN_us':>10}{'speedup':>9}")
    for sh in shapes:
        pd._FORCE_BH = 1
        t1 = measure(dev, torch.randn(*sh))
        pd._FORCE_BH = None
        tN = measure(dev, torch.randn(*sh))
        W = sh[-1]
        Wt = (W + 31) // 32
        Ht = 1
        for d in sh[:-1]:
            Ht *= d
        Ht //= 32
        cfg = pd._resolve_compute_config(None)
        bh = pd._regime_a_block_height(Ht, min(Ht, 64), False, 0, Wt, ttnn.bfloat16, True, cfg)
        print(f"{str(sh):22}{bh:>4}{t1/1000:>10.2f}{tN/1000:>10.2f}{(t1/tN):>8.2f}x")
    print("RESULTS_END")
finally:
    pd._FORCE_BH = None
    ttnn.close_device(dev)
