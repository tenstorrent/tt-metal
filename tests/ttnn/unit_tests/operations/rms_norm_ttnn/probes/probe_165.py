"""Perf round 2, Step 1 -- FOCUS = (1,1,8192,2304) INTERLEAVED (worst ratio 0.908).

Three questions, one device session:
  Q1  what plan/blocking does the focus shape resolve to, and on how many cores?
  Q2  what is the DRAM read+write roofline for this exact tensor pair? (ttnn.clone
      reference: pure interleaved DRAM read + write, no compute)
  Q3  is the shape LOAD-IMBALANCE bound rather than DRAM bound?  Rt=256 over a
      110-core grid is 36 cores x 3 tile-rows + 74 x 2 -- an 0.776 balance factor.
      Sweep Rt around exact multiples of the grid and print effective GB/s.
      If GB/s jumps at the balanced points, the wall is imbalance, not DRAM.
"""
import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")
os.environ["RMS_TRACE_BLOCKING"] = "1"

import torch, ttnn
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn

K = "DEVICE KERNEL DURATION [ns]"


def ns(device):
    ttnn.ReadDeviceProfiler(device)
    tot, found = 0.0, False
    for progs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for p in progs:
            e = (getattr(p, "program_analyses_results", None) or {}).get(K)
            if e is not None:
                tot += float(e.duration)
                found = True
    return tot if found else float("nan")


dev = ttnn.open_device(device_id=0)
try:
    g = dev.compute_with_storage_grid_size()
    NC = g.x * g.y
    print(f"RESULT grid = {g.x} x {g.y} = {NC} cores")

    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False

    def run_case(rows, W, label, do_clone=False):
        torch.manual_seed(0)
        tx = torch.randn(1, 1, rows, W, dtype=torch.float32).to(torch.bfloat16)
        x = ttnn.from_torch(
            tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tg = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
        gm = ttnn.from_torch(tg, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        kw = dict(epsilon=1e-12, compute_kernel_config=cfg, weight=gm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        o = rms_norm_ttnn(x, **kw)
        ttnn.deallocate(o)
        ttnn.synchronize_device(dev)
        ns(dev)
        o = rms_norm_ttnn(x, **kw)
        ttnn.synchronize_device(dev)
        t = ns(dev)
        ttnn.deallocate(o)
        bytes_moved = 2 * rows * W * 2
        gbs = bytes_moved / t  # bytes/ns == GB/s
        rt = rows // 32
        q, r = divmod(rt, NC)
        used = min(rt, NC)
        rmax = q + (1 if r else 0)
        bal = (rt / (used * rmax)) if rmax else 1.0
        print(
            f"RESULT {label:34s} rows={rows:6d} W={W:5d} Rt={rt:4d} ns={t:9.0f} "
            f"GB/s={gbs:6.1f} rowmax={rmax} bal={bal:.3f}"
        )
        if do_clone:
            c = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(c)
            ttnn.synchronize_device(dev)
            ns(dev)
            c = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.synchronize_device(dev)
            tc = ns(dev)
            ttnn.deallocate(c)
            print(
                f"RESULT {'  clone(same tensor)':34s} rows={rows:6d} W={W:5d} "
                f"ns={tc:9.0f} GB/s={bytes_moved/tc:6.1f}"
            )
        ttnn.deallocate(x)
        ttnn.deallocate(gm)

    # Q1+Q2: the focus shape and the runner-up, each against its clone roofline
    run_case(8192, 2304, "FOCUS 8192x2304", do_clone=True)
    run_case(8192, 1024, "RUNNER-UP 8192x1024", do_clone=True)

    # Q3: balance sweep at W=2304.  Rt = k*NC is perfectly balanced.
    for rt in (2 * NC, 2 * NC + 1, int(2.5 * NC), 3 * NC - 1, 3 * NC, 256, 3 * NC + 1):
        run_case(rt * 32, 2304, f"balance-sweep Rt={rt}")
finally:
    ttnn.close_device(dev)
