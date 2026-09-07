"""Perf 3, Step 1 -- does the focus family's wall track total BYTES or ROWMAX?

W = 1024 fixed (the focus shape's width, wt_per_core = 32, BLOCK_ROWS = 2), sweeping
Rt so the row-split balance factor Rt/(110*ceil(Rt/110)) moves independently of the
byte count.  If GB/s is flat across balance, the wall is aggregate-DRAM bound and a
"balance the split" idea is roofline-gated; if GB/s tracks balance, it is not.
"""
import os
import statistics

for k, v in (
    ("TT_METAL_DEVICE_PROFILER", "1"),
    ("TT_METAL_PROFILER_MID_RUN_DUMP", "1"),
    ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
    ("TT_METAL_LOGGER_LEVEL", "error"),
):
    os.environ.setdefault(k, v)
import ttnn  # noqa: E402
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn  # noqa: E402

K = "DEVICE KERNEL DURATION [ns]"
CORES = 110


def main():
    import torch

    def ns(dev):
        ttnn.ReadDeviceProfiler(dev)
        t, f = 0.0, False
        for progs in (ttnn.get_latest_programs_perf_data() or {}).values():
            for p in progs:
                e = (getattr(p, "program_analyses_results", None) or {}).get(K)
                if e is not None:
                    t += float(e.duration)
                    f = True
        return t if f else float("nan")

    W = int(os.environ.get("RMS_W", "1024"))
    RTS = [int(t) for t in os.environ.get("RMS_RTS", "110,220,240,256,275,330,352").split(",")]
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    dev = ttnn.open_device(device_id=0)
    try:
        torch.manual_seed(1)
        tg = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
        gm = ttnn.from_torch(tg, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        for rt in RTS:
            rows = rt * 32
            shape = (1, 1, rows, W)
            torch.manual_seed(0)
            tx = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
            x = ttnn.from_torch(
                tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            kw = dict(epsilon=1e-12, compute_kernel_config=cfg, weight=gm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            o = rms_norm_ttnn(x, **kw)
            ttnn.synchronize_device(dev)
            ttnn.deallocate(o)
            ns(dev)
            s = []
            for _ in range(3):
                o = rms_norm_ttnn(x, **kw)
                ttnn.synchronize_device(dev)
                v = ns(dev)
                ttnn.deallocate(o)
                if v == v:
                    s.append(v)
            t = statistics.median(s)
            rowmax = -(-rt // CORES)
            bal = rt / (CORES * rowmax)
            gbs = rows * W * 2 * 2 / t  # bytes/ns == GB/s
            print(
                f"RESULT rt={rt:4d} rows={rows:6d} rowmax={rowmax} bal={bal:5.3f} "
                f"ns={t:9.0f} GB/s={gbs:6.1f}",
                flush=True,
            )
            ttnn.deallocate(x)
    finally:
        ttnn.close_device(dev)


main()
