import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import statistics, sys
import torch, ttnn
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD
from eval.sharding import shard_config, auto_shard_config

_ML = ttnn.TensorMemoryLayout
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
N_WARMUP = 2
N_TRIALS = int(os.environ.get("RMS_TRIALS", "5"))
N_REPS = int(os.environ.get("RMS_REPS", "3"))


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


# name: shape, shard|"auto"|None, memory_layout, mode, fp32_dest, ceiling_ns
CASES = {
    # ---- the refinement's TARGETS: interleaved prefill --------------------
    "P1_int1024_g": ((1, 1, 8192, 1024), None, _ML.INTERLEAVED, "gamma", False, 89992),
    "P2_int1024_gb": ((1, 1, 8192, 1024), None, _ML.INTERLEAVED, "gamma_bias", False, 89992),
    "P3_int1024_n": ((1, 1, 8192, 1024), None, _ML.INTERLEAVED, "no_gamma", False, 89992),
    "P4_int2048_g": ((1, 1, 8192, 2048), None, _ML.INTERLEAVED, "gamma", False, 0),
    "P5_int5120_gbr": ((1, 1, 8192, 5120), None, _ML.INTERLEAVED, "gamma_bias_residual", False, 0),
    "P6_int7168_g": ((1, 1, 8192, 7168), None, _ML.INTERLEAVED, "gamma", False, 589591),
    # ---- the STREAM regime (x re-read in pass B): Lamp L-RES-FUSE's only home ----
    "S1_stream_gbr": ((1, 1, 1024, 16384), None, _ML.INTERLEAVED, "gamma_bias_residual", False, 0),
    "S2_stream_r": ((1, 1, 1024, 16384), None, _ML.INTERLEAVED, "residual", False, 0),
    # ---- guards: combine / sharded paths (must not regress) ---------------
    "G1_w7168_g28": ((1, 1, 32, 7168), ([32, 256], (7, 4)), _ML.WIDTH_SHARDED, "gamma", False, 5481),
    "G2_w5120_gbr": ((1, 1, 32, 5120), ([32, 160], (8, 4)), _ML.WIDTH_SHARDED, "gamma_bias_residual", True, 6555),
    "G3_blk8192": ((1, 1, 8192, 1024), ([1024, 128], (8, 8)), _ML.BLOCK_SHARDED, "gamma", False, 28619),
    "G4_blk7168": ((1, 1, 7168, 1024), ([896, 128], (8, 8)), _ML.BLOCK_SHARDED, "gamma_bias_residual", False, 34569),
    "G5_int7168_ws": ((1, 1, 32, 7168), None, _ML.INTERLEAVED, "gamma", False, 14894),
    "G6_band512_rm": ((1, 1, 256, 512), "auto", _ML.WIDTH_SHARDED, "gamma", False, 20000),
}


def build(device, name):
    shape, shard, ml, mode, fp32_dest, ceiling = CASES[name]
    rm = name.endswith("_rm")
    torch.manual_seed(0)
    W = shape[-1]
    tx = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    lay = ttnn.ROW_MAJOR_LAYOUT if rm else ttnn.TILE_LAYOUT
    if shard == "auto":
        mc = auto_shard_config(list(shape), ml, layout=lay, dtype=ttnn.bfloat16, device=device)
    elif shard is not None:
        mc = shard_config(shard[0], shard[1], ml, layout=lay, dtype=ttnn.bfloat16, device=device)
    else:
        mc = ttnn.DRAM_MEMORY_CONFIG
    x = ttnn.from_torch(tx, dtype=ttnn.bfloat16, layout=lay, device=device, memory_config=mc)
    kwargs = {"epsilon": 1e-12, "compute_kernel_config": _cfg(fp32_dest), "memory_config": x.memory_config()}
    ref = {"input_tensor": tx.float()}

    def _vec(seed):
        torch.manual_seed(seed)
        t = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
        return t, ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=lay, device=device)

    if "gamma" in mode and mode != "no_gamma":
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
            tr, dtype=ttnn.bfloat16, layout=lay, device=device, memory_config=mc
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


def relrms(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    return float(((a - b).pow(2).mean().sqrt()) / (b.pow(2).mean().sqrt() + 1e-30))


def measure(device, name):
    run, expected, ceiling, live = build(device, name)
    out = run()
    got = ttnn.to_torch(out)
    p, r = pcc(got, expected), relrms(got, expected)
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
    return ns, p, r


def sweep(variants, names=None):
    """variants: list of (label, {PD attr: value})"""
    names = names or list(CASES)
    RES = {}
    device = ttnn.open_device(device_id=0)
    saved = {}
    try:
        for rep in range(N_REPS):
            for label, knobs in variants:
                for k, v in knobs.items():
                    saved.setdefault(k, getattr(PD, k))
                    setattr(PD, k, v)
                for name in names:
                    ns, p, r = measure(device, name)
                    RES.setdefault((name, label), []).append((ns, p, r))
                for k, v in saved.items():
                    setattr(PD, k, v)
        base_label = variants[0][0]
        hdr = f"{'case':16s}" + "".join(f"{l:>11s}" for l, _ in variants)
        print("RESULT " + hdr)
        for name in names:
            row = f"{name:16s}"
            for label, _ in variants:
                row += f"{min(x[0] for x in RES[(name, label)]):11.0f}"
            print("RESULT " + row)
        print("RESULT --- speedup vs " + base_label + " (>1 = faster) ---")
        for name in names:
            b = min(x[0] for x in RES[(name, base_label)])
            row = f"{name:16s}"
            for label, _ in variants:
                row += f"{b / min(x[0] for x in RES[(name, label)]):11.3f}"
            pcs = " ".join(f"{l}:{min(x[1] for x in RES[(name,l)]):.6f}" for l, _ in variants)
            print("RESULT " + row + "  pcc " + pcs)
    finally:
        ttnn.close_device(device)
    return RES
