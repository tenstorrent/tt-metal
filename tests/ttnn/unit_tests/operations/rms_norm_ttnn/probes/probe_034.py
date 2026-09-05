import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import statistics
import torch
import ttnn

from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD
from eval.sharding import shard_config

_ML = ttnn.TensorMemoryLayout
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
N_WARMUP = 2
N_TRIALS = int(os.environ.get("RMS_TRIALS", "5"))


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


def _cfg(fp32_dest):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = fp32_dest
    c.math_approx_mode = False
    return c


# name, shape, (shard, grid) or None, memory_layout, mode, fp32_dest, ceiling_ns
CASES = {
    # --- the three Refinement-1 targets ---
    "A_w7168_g28": ((1, 1, 32, 7168), ([32, 256], (7, 4)), _ML.WIDTH_SHARDED, "gamma", False, 5481),
    "B_w5120_gbr": ((1, 1, 32, 5120), ([32, 160], (8, 4)), _ML.WIDTH_SHARDED, "gamma_bias_residual", True, 6555),
    "C_w5120_g32": ((1, 1, 32, 5120), ([32, 160], (8, 4)), _ML.WIDTH_SHARDED, "gamma", False, 5267),
    # --- guards on the same combine path (must not regress) ---
    "D_blk8192": ((1, 1, 8192, 1024), ([1024, 128], (8, 8)), _ML.BLOCK_SHARDED, "gamma", False, 28619),
    "E_blk7168": ((1, 1, 7168, 1024), ([896, 128], (8, 8)), _ML.BLOCK_SHARDED, "gamma_bias_residual", False, 34569),
    "F_w1024_g8": ((1, 1, 32, 1024), ([32, 128], (8, 1)), _ML.WIDTH_SHARDED, "gamma", False, 4110),
    "G_w2304_g9": ((1, 1, 32, 2304), ([32, 256], (9, 1)), _ML.WIDTH_SHARDED, "gamma", False, 4617),
    # --- interleaved width-split guards (combine path too) ---
    "J_w4800_g30": ((1, 1, 32, 4800), ([32, 160], (10, 3)), _ML.WIDTH_SHARDED, "gamma", False, 5000),
    "K_w5120_g40": ((1, 1, 32, 5120), ([32, 128], (10, 4)), _ML.WIDTH_SHARDED, "gamma", False, 5267),
    "L_w8192_g64": ((1, 1, 32, 8192), ([32, 128], (8, 8)), _ML.WIDTH_SHARDED, "gamma", False, 6000),
    "H_int7168": ((1, 1, 32, 7168), None, _ML.INTERLEAVED, "gamma", False, 14894),
    "I_int5120": ((1, 1, 32, 5120), None, _ML.INTERLEAVED, "gamma", False, 75825),
}


def build(device, name):
    shape, shard, ml, mode, fp32_dest, ceiling = CASES[name]
    torch.manual_seed(0)
    W = shape[-1]
    tx = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    if shard is not None:
        mc = shard_config(shard[0], shard[1], ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    else:
        mc = ttnn.DRAM_MEMORY_CONFIG
    x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    kwargs = {"epsilon": 1e-12, "compute_kernel_config": _cfg(fp32_dest), "memory_config": x.memory_config()}
    ref = {"input_tensor": tx.float()}

    def _vec(seed):
        torch.manual_seed(seed)
        t = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
        return t, ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

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
        kwargs["residual_input_tensor"] = ttnn.from_torch(
            tr, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc
        )
        ref["residual_input_tensor"] = tr.float()
    expected = torch_rms_norm_ttnn(
        ref["input_tensor"],
        epsilon=1e-12,
        weight=ref.get("weight"),
        bias=ref.get("bias"),
        residual_input_tensor=ref.get("residual_input_tensor"),
    )
    live = [x] + [v for v in kwargs.values() if isinstance(v, ttnn.Tensor)]
    return (lambda: rms_norm_ttnn(x, **kwargs)), expected, ceiling, live


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def measure(device, name):
    run, expected, ceiling, live = build(device, name)
    out = run()
    got = ttnn.to_torch(out)
    p = pcc(got, expected)
    del out, got
    for _ in range(N_WARMUP):
        run()
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
    return ns, p, ceiling, samples


TARGETS = list(CASES.keys())
device = ttnn.open_device(device_id=0)
try:
    for rep in range(2):
        for name in TARGETS:
            ns, p, ceil, samples = measure(device, name)
            print(f"S5 rep{rep} gated {name:14s} ns={ns:8.0f} r={ns/ceil:6.3f} pcc={p:.6f}")
finally:
    ttnn.close_device(device)
