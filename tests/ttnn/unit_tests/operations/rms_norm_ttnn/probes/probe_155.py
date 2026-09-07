"""Perf round 1, Step 1 -- rank the `perf` loose-case group by measured/achievable.

No LOOSE_CASES entry for this op carries an `attention:` note, so the coordinator's
step-1 fallback applies: measure EVERY case in the `perf` group, divide each measured
device-kernel-ns by that case's own clock-scaled `achievable_ns` (further divided by
`minimum_expected_speedup` when present), and take the largest ratio.

Reads the cases straight out of eval/golden_tests/rms_norm_ttnn/feature_spec.py so the
ranking cannot drift from the spec.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import statistics
import sys

import torch
import ttnn

from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn
from eval.sharding import shard_config
from eval.golden_tests.rms_norm_ttnn import feature_spec as FS

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
N_TRIALS = int(os.environ.get("RMS_TRIALS", "3"))


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def perf_cases():
    out = []
    for c in FS.LOOSE_CASES:
        if c.get("group") != "perf":
            continue
        out.append(c)
    return out


def _label(i, c):
    shape = c["inputs"][0]
    ml = str(c["memory_layout"]).split(".")[-1][:5]
    ex = c["extras"]
    g = c.get("gamma_mode", "gamma")
    f32 = "T" if c.get("fp32_dest_acc_en") else "F"
    gl = "RM" if c.get("gamma_layout") == ttnn.ROW_MAJOR_LAYOUT else "TI"
    return f"{i:02d} {shape[-2]}x{shape[-1]} {ml} {g[:18]} f32{f32} w{gl}"


def build(device, c):
    shape = list(c["inputs"][0])
    W = shape[-1]
    ex = c["extras"]
    dtype = c["dtype"]
    lay = c["layout"]
    torch.manual_seed(0)
    tx = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    ml = c["memory_layout"]
    if "shard_shape" in ex:
        mc = shard_config(ex["shard_shape"], ex["core_grid"], ml, layout=lay, dtype=dtype, device=device)
    else:
        mc = ttnn.DRAM_MEMORY_CONFIG
    x = ttnn.from_torch(tx, dtype=dtype, layout=lay, device=device, memory_config=mc)

    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ex.get("math_fidelity", ttnn.MathFidelity.HiFi2)
    cfg.fp32_dest_acc_en = bool(c.get("fp32_dest_acc_en"))
    cfg.math_approx_mode = False

    kwargs = {"epsilon": 1e-12, "compute_kernel_config": cfg, "memory_config": x.memory_config()}
    ref = {"input_tensor": tx.float()}
    mode = c.get("gamma_mode", "gamma")
    glay = c.get("gamma_layout", ttnn.TILE_LAYOUT)
    gdt = c.get("gamma_dtype", ttnn.bfloat16)

    def _vec(seed):
        torch.manual_seed(seed)
        t = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
        return t, ttnn.from_torch(t, dtype=gdt, layout=glay, device=device)

    if "gamma" in mode:
        t, v = _vec(1)
        kwargs["weight"] = v
        ref["weight"] = t.float()
    if "bias" in mode:
        t, v = _vec(2)
        kwargs["bias"] = v
        ref["bias"] = t.float()
    if "residual" in mode:
        torch.manual_seed(3)
        tr = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        kwargs["residual_input_tensor"] = ttnn.from_torch(tr, dtype=dtype, layout=lay, device=device, memory_config=mc)
        ref["residual_input_tensor"] = tr.float()

    expected = torch_rms_norm_ttnn(
        ref["input_tensor"],
        epsilon=1e-12,
        weight=ref.get("weight"),
        bias=ref.get("bias"),
        residual_input_tensor=ref.get("residual_input_tensor"),
    )
    live = [x] + [v for v in kwargs.values() if isinstance(v, ttnn.Tensor)]
    return (lambda: rms_norm_ttnn(x, **kwargs)), expected, live


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def main():
    device = ttnn.open_device(device_id=0)
    rows = []
    try:
        for i, c in enumerate(perf_cases()):
            lbl = _label(i, c)
            try:
                run, expected, live = build(device, c)
                out = run()
                got = ttnn.to_torch(out)
                p = pcc(got, expected)
                del out, got
                ttnn.synchronize_device(device)
                _read_kernel_ns(device)
                samples = []
                for _ in range(N_TRIALS):
                    run()
                    ttnn.synchronize_device(device)
                    v = _read_kernel_ns(device)
                    if v is not None:
                        samples.append(v)
                ns = statistics.median(samples) if samples else float("nan")
                for t in live:
                    try:
                        ttnn.deallocate(t)
                    except Exception:
                        pass
            except Exception as e:
                print(f"RESULT {lbl}  ERROR {type(e).__name__}: {e}")
                continue
            ex = c["extras"]
            ach = float(ex["achievable_ns"])
            ceil_ns = ach / float(ex.get("minimum_expected_speedup", 1.0))
            rows.append((ns / ceil_ns, lbl, ns, ceil_ns, p))
            print(f"RESULT {lbl}  ns={ns:10.0f}  ceil={ceil_ns:10.0f}  ratio={ns/ceil_ns:6.3f}  pcc={p:.6f}")
    finally:
        ttnn.close_device(device)
    print("RESULT ==== ranked worst-first (measured / clock-scaled ceiling) ====")
    for r, lbl, ns, ce, p in sorted(rows, reverse=True):
        print(f"RESULT   {r:6.3f}  {lbl}   {ns:.0f} vs {ce:.0f}")


main()
