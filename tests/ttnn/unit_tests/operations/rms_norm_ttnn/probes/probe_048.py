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
from eval.sharding import shard_config, auto_shard_config

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


# name: shape, (shard, grid)|"auto"|None, memory_layout, mode, fp32_dest, ceiling_ns
CASES = {
    # --- the three Refinement-1 targets (BLOCK_ROWS == 1, identity branch) ---
    "A_w7168_g28": ((1, 1, 32, 7168), ([32, 256], (7, 4)), _ML.WIDTH_SHARDED, "gamma", False, 5481),
    "B_w5120_gbr": ((1, 1, 32, 5120), ([32, 160], (8, 4)), _ML.WIDTH_SHARDED, "gamma_bias_residual", True, 6555),
    "C_w5120_g32": ((1, 1, 32, 5120), ([32, 160], (8, 4)), _ML.WIDTH_SHARDED, "gamma", False, 5267),
    # --- the two 64-core BLOCK shards (COMPACT branch, multi-round) ---
    "D_blk8192": ((1, 1, 8192, 1024), ([1024, 128], (8, 8)), _ML.BLOCK_SHARDED, "gamma", False, 28619),
    "E_blk7168": ((1, 1, 7168, 1024), ([896, 128], (8, 8)), _ML.BLOCK_SHARDED, "gamma_bias_residual", False, 34569),
    # --- combine-path guards ---
    "F_w1024_g8": ((1, 1, 32, 1024), ([32, 128], (8, 1)), _ML.WIDTH_SHARDED, "gamma", False, 4110),
    "G_w2304_g9": ((1, 1, 32, 2304), ([32, 256], (9, 1)), _ML.WIDTH_SHARDED, "gamma", False, 4617),
    "L_w8192_g64": ((1, 1, 32, 8192), ([32, 128], (8, 8)), _ML.WIDTH_SHARDED, "gamma", False, 6000),
    "P_wcompact": ((1, 1, 1024, 512), ([1024, 128], (4, 1)), _ML.WIDTH_SHARDED, "gamma", False, 20000),
    # --- interleaved width split (streamed combine) ---
    "H_int7168": ((1, 1, 32, 7168), None, _ML.INTERLEAVED, "gamma", False, 14894),
    # --- non-combine guards ---
    "I_int_pre": ((1, 1, 8192, 1024), None, _ML.INTERLEAVED, "gamma", False, 89992),
    "R_band512_rm": ((1, 1, 256, 512), "auto", _ML.WIDTH_SHARDED, "gamma", False, 20000),
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


# (name, fin_spread, fire_and_forget, mcast_faces)
VARIANTS = [
    ("base", False, False, 0),
    ("ff", False, True, 0),
    ("m3", False, False, 3),
    ("ff_m3", False, True, 3),
]

RES = {}
device = ttnn.open_device(device_id=0)
NAMES = list(CASES)
try:
    for rep in range(3):
        for vname, spread, ff, mf in VARIANTS:
            PD.COMBINE_FIN_SPREAD = spread
            PD.COMBINE_MCAST_FIRE_AND_FORGET = ff
            PD.COMBINE_MCAST_FACES = mf
            for name in NAMES:
                ns, p, r = measure(device, name)
                RES.setdefault((name, vname), []).append((ns, p, r))
    hdr = f"{'case':14s}" + "".join(f"{v:>10s}" for v, _, _, _ in VARIANTS) + f"{'ceil':>9s}"
    print("RESULT " + hdr)
    for name in NAMES:
        row = f"{name:14s}"
        for vname, _, _, _ in VARIANTS:
            row += f"{min(x[0] for x in RES[(name, vname)]):10.0f}"
        row += f"{CASES[name][5]:9d}"
        print("RESULT " + row)
    print("RESULT --- ratio vs base (>1 = faster), pcc/relrms of each variant ---")
    for name in NAMES:
        b = min(x[0] for x in RES[(name, "base")])
        row = f"{name:14s}"
        for vname, _, _, _ in VARIANTS:
            a = min(x[0] for x in RES[(name, vname)])
            row += f"{b/a:10.3f}"
        pcs = " ".join(
            f"{v}:{min(x[1] for x in RES[(name,v)]):.6f}/{max(x[2] for x in RES[(name,v)]):.5f}"
            for v, _, _, _ in VARIANTS
        )
        print("RESULT " + row + "  " + pcs)
finally:
    ttnn.close_device(device)
