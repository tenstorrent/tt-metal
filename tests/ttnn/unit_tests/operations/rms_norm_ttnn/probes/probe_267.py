"""Perf 3, Step 1 — measure the focus shape (and its neighbours) once, fresh cache.

Device kernel time has no warm-up transient, so this takes ONE reading per case by
default (RMS_READS raises it only when a small-shape spread needs bounding).
Used for the cumulative-peel ablation: the peel driver seds the RMS_ABLATE_* defines
in the shipped kernel heads, runs this, and restores.
"""
import os, statistics, sys

for k, v in (
    ("TT_METAL_DEVICE_PROFILER", "1"),
    ("TT_METAL_PROFILER_MID_RUN_DUMP", "1"),
    ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
    ("TT_METAL_LOGGER_LEVEL", "error"),
):
    os.environ.setdefault(k, v)
import ttnn  # noqa: E402
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn


def main():
    import torch  # local: ttnn/ forbids a module-scope torch import

    K = "DEVICE KERNEL DURATION [ns]"
    READS = int(os.environ.get("RMS_READS", "1"))
    TAG = os.environ.get("RMS_TAG", "run")

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

    # label, shape, gamma_mode, fp32_dest
    ALL = {
        "focus_8192x1024": ((1, 1, 8192, 1024), "gamma", False),
        "c05_8192x2304": ((1, 1, 8192, 2304), "gamma", False),
        "c06_8192x5120": ((1, 1, 8192, 5120), "gamma", False),
        "c00_32x1024": ((1, 1, 32, 1024), "gamma", False),
    }
    want = os.environ.get("RMS_CASES", "focus_8192x1024").split(",")

    dev = ttnn.open_device(device_id=0)
    try:
        for label in want:
            shape, mode, f32 = ALL[label]
            W = shape[-1]
            cfg = ttnn.ComputeConfigDescriptor()
            cfg.math_fidelity = ttnn.MathFidelity.HiFi2
            cfg.fp32_dest_acc_en = f32
            cfg.math_approx_mode = False
            torch.manual_seed(0)
            tx = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
            x = ttnn.from_torch(
                tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            kw = dict(epsilon=1e-12, compute_kernel_config=cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ref = dict(input_tensor=tx.float())
            torch.manual_seed(1)
            tg = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
            kw["weight"] = ttnn.from_torch(tg, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            ref["weight"] = tg.float()
            if "bias" in mode:
                torch.manual_seed(2)
                tb = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
                kw["bias"] = ttnn.from_torch(tb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
                ref["bias"] = tb.float()
            exp = torch_rms_norm_ttnn(**ref, epsilon=1e-12)
            out = rms_norm_ttnn(x, **kw)
            p = pcc(ttnn.to_torch(out), exp)
            ttnn.deallocate(out)
            ttnn.synchronize_device(dev)
            ns(dev)
            s = []
            for _ in range(READS):
                o = rms_norm_ttnn(x, **kw)
                ttnn.synchronize_device(dev)
                v = ns(dev)
                ttnn.deallocate(o)
                if v == v:
                    s.append(v)
            m = statistics.median(s) if s else float("nan")
            print(f"RESULT {TAG} {label} ns={m:.0f} min={min(s):.0f} n={len(s)} pcc={p:.6f}", flush=True)
    finally:
        ttnn.close_device(dev)


main()
