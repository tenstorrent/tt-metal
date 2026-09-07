"""Perf 2 final: the ROW_MAJOR-ACTIVATION plans the D40 guard fix newly ADMITS to the
broadcast.  Correctness is proved by golden; this is the no-regression half."""
import os, statistics

for k, v in (
    ("TT_METAL_DEVICE_PROFILER", "1"),
    ("TT_METAL_PROFILER_MID_RUN_DUMP", "1"),
    ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
    ("TT_METAL_LOGGER_LEVEL", "error"),
):
    os.environ.setdefault(k, v)
import torch, ttnn
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn

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


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


CASES = [
    ("RM act, TILE weight  (1,1,64,128)", (1, 1, 64, 128)),
    ("RM act, TILE weight  (1,1,1024,2048)", (1, 1, 1024, 2048)),
    ("RM act, TILE weight  (1,1,8192,1024)", (1, 1, 8192, 1024)),
]
dev = ttnn.open_device(device_id=0)
try:
    for label, shape in CASES:
        W = shape[-1]
        cfg = ttnn.ComputeConfigDescriptor()
        cfg.math_fidelity = ttnn.MathFidelity.HiFi2
        cfg.fp32_dest_acc_en = False
        cfg.math_approx_mode = False
        torch.manual_seed(0)
        tx = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        x = ttnn.from_torch(
            tx, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        torch.manual_seed(1)
        tg = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
        gm = ttnn.from_torch(tg, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        kw = dict(epsilon=1e-12, compute_kernel_config=cfg, weight=gm, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        exp = torch_rms_norm_ttnn(tx.float(), epsilon=1e-12, weight=tg.float())
        o = rms_norm_ttnn(x, **kw)
        p = pcc(ttnn.to_torch(o), exp)
        ttnn.deallocate(o)
        ttnn.synchronize_device(dev)
        ns(dev)
        r = []
        for _ in range(5):
            o = rms_norm_ttnn(x, **kw)
            ttnn.synchronize_device(dev)
            r.append(ns(dev))
            ttnn.deallocate(o)
        print(f"RESULT {label:34s} min={min(r):9.0f} med={statistics.median(r):9.0f} pcc={p:.6f}")
        ttnn.deallocate(x)
        ttnn.deallocate(gm)
finally:
    ttnn.close_device(dev)
