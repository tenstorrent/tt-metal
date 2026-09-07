"""Round 2, Step 1b -- what does the PER-CHANNEL operand cost on the interleaved
row-split prefill, and which regime does the wide-residual case land in?

The focus shape (1,1,8192,2304) INTERLEAVED runs on 110 cores and EVERY core reads
the WHOLE gamma row (72 tiles x 2048 B = 147 KB).  That is 110 x 147 KB = 16.2 MB of
DRAM traffic against a 75.5 MB payload -- 21% on top.  Measure it by varying only the
operand count (none / gamma / gamma+bias), everything else identical.
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


dev = ttnn.open_device(device_id=0)
try:

    def go(rows, W, mode, f32, label):
        cfg = ttnn.ComputeConfigDescriptor()
        cfg.math_fidelity = ttnn.MathFidelity.HiFi2
        cfg.fp32_dest_acc_en = f32
        cfg.math_approx_mode = False
        torch.manual_seed(0)
        tx = torch.randn(1, 1, rows, W, dtype=torch.float32).to(torch.bfloat16)
        x = ttnn.from_torch(
            tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        kw = dict(epsilon=1e-12, compute_kernel_config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        live = [x]

        def vec(s):
            torch.manual_seed(s)
            t = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
            v = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            live.append(v)
            return v

        ncross = 2  # x read + y write
        if "gamma" in mode:
            kw["weight"] = vec(1)
        if "bias" in mode:
            kw["bias"] = vec(2)
        if "residual" in mode:
            torch.manual_seed(3)
            tr = torch.randn(1, 1, rows, W, dtype=torch.float32).to(torch.bfloat16)
            r = ttnn.from_torch(
                tr, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            kw["residual_input_tensor"] = r
            live.append(r)
            ncross = 3
        o = rms_norm_ttnn(x, **kw)
        ttnn.deallocate(o)
        ttnn.synchronize_device(dev)
        ns(dev)
        o = rms_norm_ttnn(x, **kw)
        ttnn.synchronize_device(dev)
        t = ns(dev)
        ttnn.deallocate(o)
        payload = ncross * rows * W * 2
        print(f"RESULT {label:38s} ns={t:10.0f}  payload_GB/s={payload/t:6.1f}")
        for v in live:
            try:
                ttnn.deallocate(v)
            except Exception:
                pass
        return t

    print("RESULT === FOCUS (1,1,8192,2304) INTER: operand-count sweep ===")
    t0 = go(8192, 2304, "none", False, "8192x2304 no operand")
    t1 = go(8192, 2304, "gamma", False, "8192x2304 gamma        (FOCUS)")
    t2 = go(8192, 2304, "gamma_bias", False, "8192x2304 gamma+bias")
    print(
        f"RESULT   per-operand delta: gamma +{t1-t0:.0f} ns ({(t1/t0-1)*100:.1f}%), "
        f"bias +{t2-t1:.0f} ns ({(t2/t1-1)*100:.1f}%)"
    )

    print("RESULT === RUNNER-UP (1,1,8192,1024) INTER: operand-count sweep ===")
    u0 = go(8192, 1024, "none", False, "8192x1024 no operand")
    u1 = go(8192, 1024, "gamma", False, "8192x1024 gamma")
    print(f"RESULT   gamma delta +{u1-u0:.0f} ns ({(u1/u0-1)*100:.1f}%)")

    print("RESULT === wide interleaved cases (regime check) ===")
    go(8192, 5120, "gamma", False, "8192x5120 gamma")
    go(8192, 7168, "gamma", False, "8192x7168 gamma")
    go(8192, 7168, "gamma_bias_residual", True, "8192x7168 gbr f32T   (#15)")
    go(8192, 5120, "gamma_bias_residual", True, "8192x5120 gbr f32T   (#14)")
finally:
    ttnn.close_device(dev)
