"""Round 2, Step 1 -- CUMULATIVE ablation peel on the focus shape.

Focus = (1,1,8192,2304) INTERLEAVED DRAM, bf16, gamma, HiFi2, fp32_dest_acc_en=False
(worst measured/achievable ratio of the `perf` group: 0.908).

Run once per RMS_ABLATE_* define set (the defines live at the head of each kernel
and are toggled by the caller); the stages are peeled CUMULATIVELY, never one at a
time, because the reader's NoC reads overlap the writer's NoC writes and the TRISC
compute -- removing one alone lets its partner fill the gap.

Correctness is meaningless with any switch on; only the ns matter.
"""
import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import torch, ttnn
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn

K = "DEVICE KERNEL DURATION [ns]"
TAG = os.environ.get("RMS_ABLATE_TAG", "?")


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
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    for rows, W, label in ((8192, 2304, "FOCUS 8192x2304"), (8192, 1024, "8192x1024")):
        torch.manual_seed(0)
        tx = torch.randn(1, 1, rows, W, dtype=torch.float32).to(torch.bfloat16)
        x = ttnn.from_torch(
            tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        torch.manual_seed(1)
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
        print(f"RESULT ABLATE[{TAG}] {label:20s} ns={t:10.0f}")
        ttnn.deallocate(x)
        ttnn.deallocate(gm)
finally:
    ttnn.close_device(dev)
