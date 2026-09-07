"""Perf 3 idea B -- pass A's DEST-fold CEILING, and the GROUPED fold that replaces it.

WHAT IS BEING COMPARED (every variant at the op's pinned, USER-OWNED precision config:
each case's own math_fidelity / fp32_dest_acc_en / math_approx_mode / dtypes, UNTOUCHED):

  base      the shipped op.  DEST_ACC_SQUARE_MAX_WT = 8, so the square folds the WHOLE
            chunk into DEST only when WT_CHUNK <= 8; every prefill profile (WT_CHUNK
            32..80) packs WT_CHUNK x^2 tiles per tile-row and the reduce unpacks them all.
            Serial 16-bit DEST accumulation depth: WT_CHUNK where the fold is on (<= 8).
  flat16/32/inf   the shipped FLAT fold with the ceiling raised.  Depth == WT_CHUNK.
  exact     flat_inf + CB_SQ_EXACT (charge cb_x_squared its true 1-tile width in the
            RESIDENT L1 solve, which can buy a coarser BLOCK_ROWS).
  grp4/8/16 the GROUPED fold: fold in groups of the largest divisor of WT_CHUNK that is
            <= G, packing WT_CHUNK/G tiles per tile-row.  Depth == G, so grp8 keeps the
            accumulation depth EXACTLY where the shipped ceiling already puts it while
            still deleting 7 of every 8 packs and unpacks.
  grp8x     grp8 + CB_SQ_EXACT.
  fork_off  the forked descriptor + forked kernels with the grouped fold OFF -- the
            NEUTRALITY control.  Must reproduce `base`.
  flatinf_rt  flat_inf with the reduce's per-call-width floor dropped, so per-call width 1
            keeps AccumulateViaAdd instead of falling back to ReduceTile (D20).  A
            DATAPATH probe, reported with its own pcc.

Usage:
    RMS_VARIANTS=base,grp8 RMS_CASES=focus scripts/tt-probe.sh rms_norm_ttnn < bench_fold.py
Case groups: `focus`, `perf` (the 19-case perf group), `dom` (the domain extras), `all`.
"""

import importlib
import importlib.util
import os
import statistics
import sys
from pathlib import Path

for k, v in (
    ("TT_METAL_DEVICE_PROFILER", "1"),
    ("TT_METAL_PROFILER_MID_RUN_DUMP", "1"),
    ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
    ("TT_METAL_LOGGER_LEVEL", "error"),
):
    os.environ.setdefault(k, v)

import ttnn  # noqa: E402
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD  # noqa: E402
from eval.golden_tests.rms_norm_ttnn import feature_spec as FS  # noqa: E402
from eval.sharding import shard_config  # noqa: E402

# tt-probe.sh COPIES this script into tests/.../probes/ before running it, so __file__
# is not the experiment dir.  RMS_EXP_DIR (or the absolute default) is.
HERE = Path(
    os.environ.get("RMS_EXP_DIR")
    or "/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal"
    "/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/square_fold_ceiling"
)
OP = importlib.import_module("ttnn.operations.rms_norm_ttnn.rms_norm_ttnn")
TORCH_REF = OP.torch_rms_norm_ttnn

_ML = ttnn.TensorMemoryLayout
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
N_WARMUP = int(os.environ.get("RMS_WARMUP", "1"))
N_TRIALS = int(os.environ.get("RMS_TRIALS", "3"))


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------
def _case(
    shape,
    ml=_ML.INTERLEAVED,
    shard=None,
    mode="gamma",
    fp32=False,
    layout=None,
    dtype=None,
    gamma_layout=None,
    gamma_dtype=None,
    fidelity=None,
    eps=1e-12,
):
    return dict(
        shape=tuple(shape),
        ml=ml,
        shard=shard,
        mode=mode,
        fp32=bool(fp32),
        layout=layout or ttnn.TILE_LAYOUT,
        dtype=dtype or ttnn.bfloat16,
        gamma_layout=gamma_layout or ttnn.TILE_LAYOUT,
        gamma_dtype=gamma_dtype or ttnn.bfloat16,
        fidelity=fidelity or ttnn.MathFidelity.HiFi2,
        eps=eps,
    )


PERF = {}
for i, c in enumerate(x for x in FS.LOOSE_CASES if x.get("group") == "perf"):
    ex = c["extras"]
    shape = tuple(c["inputs"][0])
    PERF[f"P{i:02d}_{shape[-2]}x{shape[-1]}"] = _case(
        shape,
        ml=c["memory_layout"],
        shard=((ex["shard_shape"], ex["core_grid"]) if "shard_shape" in ex else None),
        mode=c.get("gamma_mode", "gamma"),
        fp32=c.get("fp32_dest_acc_en", False),
        layout=c["layout"],
        dtype=c["dtype"],
        gamma_layout=c.get("gamma_layout", ttnn.TILE_LAYOUT),
        gamma_dtype=c.get("gamma_dtype", ttnn.bfloat16),
        fidelity=ex.get("math_fidelity", ttnn.MathFidelity.HiFi2),
    )
FOCUS = "P04_8192x1024"

# ---- domain extras: the regimes the perf group does not reach -------------
DOM = {
    # the fold ALREADY ships here (WT_CHUNK 4..8) -- prove no regression
    "D_w1024_8c": _case((1, 1, 32, 1024), _ML.WIDTH_SHARDED, ([32, 128], (8, 1))),
    "D_blk8192": _case((1, 1, 8192, 1024), _ML.BLOCK_SHARDED, ([1024, 128], (8, 8))),
    "D_blk7168_gbr": _case((1, 1, 7168, 1024), _ML.BLOCK_SHARDED, ([896, 128], (8, 8)), mode="gamma_bias_residual"),
    # W NON-ALIGNED: PARTIAL_W != 0, the fold is gated OFF for correctness everywhere
    "D_partialw_1000": _case((1, 1, 8192, 1000)),
    "D_partialw_2312": _case((1, 1, 8192, 2312)),
    # STREAM / ROW_RESIDENT (x re-read or only the row held): wide W, chunked
    "D_stream_16384": _case((1, 1, 1024, 16384), mode="gamma_bias_residual"),
    "D_rowres_10240": _case((1, 1, 8192, 10240), mode="gamma_bias_residual", fp32=True),
    # block-float activations, and float32 activations
    "D_bfp8_1024": _case((1, 1, 8192, 1024), dtype=ttnn.bfloat8_b, gamma_dtype=ttnn.bfloat8_b),
    "D_bfp8_5120": _case((1, 1, 8192, 5120), dtype=ttnn.bfloat8_b, gamma_dtype=ttnn.bfloat8_b),
    "D_fp32_1024": _case((1, 1, 8192, 1024), dtype=ttnn.float32, gamma_dtype=ttnn.float32),
    # ROW_MAJOR activations (tilize in pass A ahead of the square)
    "D_rm_512": _case((1, 1, 256, 512), layout=ttnn.ROW_MAJOR_LAYOUT, gamma_layout=ttnn.ROW_MAJOR_LAYOUT),
    # decode: one tile-row per core, the cross-core width combine
    "D_w7168_28c": _case((1, 1, 32, 7168), _ML.WIDTH_SHARDED, ([32, 256], (7, 4))),
    # HiFi4 + fp32 DEST -- a DIFFERENT supported precision cell, measured separately
    "D_hifi4_f32_1024": _case((1, 1, 8192, 1024), fp32=True, fidelity=ttnn.MathFidelity.HiFi4),
}

CASES = dict(PERF)
CASES.update(DOM)
GROUPS = {
    "focus": [FOCUS],
    "perf": list(PERF),
    "dom": list(DOM),
    "all": list(CASES),
}


# ---------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------
SHIPPED_FN = OP.create_program_descriptor
SHIPPED_KDIR = PD.KERNEL_DIR
_FORK = [None]


def _fork():
    if _FORK[0] is None:
        spec = importlib.util.spec_from_file_location("pd_fold", HERE / "pd_fold.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["pd_fold"] = mod
        spec.loader.exec_module(mod)
        _FORK[0] = mod
    return _FORK[0]


_KNOBS = ("DEST_ACC_SQUARE_MAX_WT", "CB_SQ_EXACT", "REDUCE_ACC_VIA_ADD_MIN_CALL_WT", "SQ_FOLD_GROUP")


def _reset():
    OP.create_program_descriptor = SHIPPED_FN
    PD.KERNEL_DIR = SHIPPED_KDIR
    for mod in (PD, _FORK[0]):
        if mod is None:
            continue
        for k in _KNOBS:
            if hasattr(mod, "_SFC_SAVED") and k in mod._SFC_SAVED:
                setattr(mod, k, mod._SFC_SAVED[k])


def _save(mod):
    if not hasattr(mod, "_SFC_SAVED"):
        mod._SFC_SAVED = {k: getattr(mod, k) for k in _KNOBS if hasattr(mod, k)}


def _shipped(kdir="k_base", **knobs):
    def apply():
        _save(PD)
        PD.KERNEL_DIR = HERE / kdir
        for k, v in knobs.items():
            setattr(PD, k, v)

    return apply


def _forked(**knobs):
    def apply():
        mod = _fork()
        _save(mod)
        OP.create_program_descriptor = mod.create_program_descriptor
        mod.KERNEL_DIR = HERE / "k_grp"
        for k, v in knobs.items():
            setattr(mod, k, v)

    return apply


VARIANTS = {
    # the honest baseline: the shipped op, kernels copied byte-for-byte to k_base so the
    # candidate's kernel path is the only thing that differs from it
    "base": _shipped(),
    # ---- the FLAT fold at a raised ceiling (no kernel change at all) --------
    "flat16": _shipped(DEST_ACC_SQUARE_MAX_WT=16),
    "flat32": _shipped(DEST_ACC_SQUARE_MAX_WT=32),
    "flatinf": _shipped(DEST_ACC_SQUARE_MAX_WT=10**9),
    "flatinf_x": _shipped(DEST_ACC_SQUARE_MAX_WT=10**9, CB_SQ_EXACT=1),
    "flatinf_rt": _shipped(DEST_ACC_SQUARE_MAX_WT=10**9, REDUCE_ACC_VIA_ADD_MIN_CALL_WT=1),
    # ---- the GROUPED fold (forked descriptor + forked kernels) -------------
    "fork_off": _forked(),
    "grp4": _forked(SQ_FOLD_GROUP=4),
    "grp8": _forked(SQ_FOLD_GROUP=8),
    "grp16": _forked(SQ_FOLD_GROUP=16),
    "grp8_x": _forked(SQ_FOLD_GROUP=8, CB_SQ_EXACT=1),
    "grp16_x": _forked(SQ_FOLD_GROUP=16, CB_SQ_EXACT=1),
    "grp8_rt": _forked(SQ_FOLD_GROUP=8, REDUCE_ACC_VIA_ADD_MIN_CALL_WT=10**9),
    # the flat fold through the FORK -- proves the fork's flat path == the shipped flat
    "fork_flatinf": _forked(DEST_ACC_SQUARE_MAX_WT=10**9),
}


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------
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


def build(device, name):
    import torch

    c = CASES[name]
    shape = list(c["shape"])
    W = shape[-1]
    lay, dt = c["layout"], c["dtype"]
    torch.manual_seed(0)
    tx = torch.randn(shape, dtype=torch.float32)
    if c["shard"] is not None:
        mc = shard_config(c["shard"][0], c["shard"][1], c["ml"], layout=lay, dtype=dt, device=device)
    else:
        mc = ttnn.DRAM_MEMORY_CONFIG
    x = ttnn.from_torch(tx, dtype=dt, layout=lay, device=device, memory_config=mc)
    # the float64 REFERENCE is taken from the tensor the device actually holds, so the
    # reported error is the KERNEL's, not the input quantization's
    ref_x = ttnn.to_torch(x).float()

    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = c["fidelity"]
    cfg.fp32_dest_acc_en = c["fp32"]
    cfg.math_approx_mode = False
    kwargs = {"epsilon": c["eps"], "compute_kernel_config": cfg, "memory_config": x.memory_config()}
    ref = {"input_tensor": ref_x}

    def _vec(seed):
        torch.manual_seed(seed)
        t = torch.randn(1, 1, 1, W, dtype=torch.float32)
        v = ttnn.from_torch(t, dtype=c["gamma_dtype"], layout=c["gamma_layout"], device=device)
        return ttnn.to_torch(v).float(), v

    mode = c["mode"]
    if "gamma" in mode:
        t, v = _vec(1)
        kwargs["weight"] = v
        ref["weight"] = t
    if "bias" in mode:
        t, v = _vec(2)
        kwargs["bias"] = v
        ref["bias"] = t
    if "residual" in mode:
        torch.manual_seed(3)
        tr = torch.randn(shape, dtype=torch.float32)
        rv = ttnn.from_torch(tr, dtype=dt, layout=lay, device=device, memory_config=mc)
        kwargs["residual_input_tensor"] = rv
        ref["residual_input_tensor"] = ttnn.to_torch(rv).float()

    expected = TORCH_REF(
        ref["input_tensor"].double(),
        epsilon=c["eps"],
        weight=ref["weight"].double() if "weight" in ref else None,
        bias=ref["bias"].double() if "bias" in ref else None,
        residual_input_tensor=ref["residual_input_tensor"].double() if "residual_input_tensor" in ref else None,
    )
    live = [x] + [v for v in kwargs.values() if isinstance(v, ttnn.Tensor)]
    return (lambda: OP.rms_norm_ttnn(x, **kwargs)), expected, live


def measure(device, name):
    run, expected, live = build(device, name)
    out = run()
    got = ttnn.to_torch(out).float()
    p, r = pcc(got, expected.float()), relrms(got, expected.float())
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
    lo = min(samples) if samples else float("nan")
    for t in live:
        try:
            ttnn.deallocate(t)
        except Exception:
            pass
    return ns, lo, p, r


def main():
    want_v = [v for v in os.environ.get("RMS_VARIANTS", "base,grp8").split(",") if v]
    grp = os.environ.get("RMS_CASES", "focus")
    names = GROUPS.get(grp) or [n for n in grp.split(",") if n]
    for v in want_v:
        assert v in VARIANTS, f"unknown variant {v}; have {list(VARIANTS)}"
    for c in names:
        assert c in CASES, f"unknown case {c}; have {list(CASES)}"

    device = ttnn.open_device(device_id=0)
    RES = {}
    try:
        for label in want_v:
            _reset()
            VARIANTS[label]()
            for name in names:
                try:
                    ns, lo, p, r = measure(device, name)
                except Exception as e:  # a variant that cannot build a case is DATA
                    print(f"RESULT {label:12s} {name:20s} ERROR {type(e).__name__}: {e}", flush=True)
                    RES[(name, label)] = None
                    continue
                RES[(name, label)] = (ns, lo, p, r)
                print(
                    f"RESULT {label:12s} {name:20s} ns={ns:10.0f} min={lo:10.0f} pcc={p:.7f} relrms={r:.5f}",
                    flush=True,
                )
    finally:
        _reset()
        ttnn.close_device(device)

    base = want_v[0]
    print(f"RESULT ==== speedup vs {base} (>1 = faster), median ns ====")
    hdr = f"{'case':20s}" + "".join(f"{v:>12s}" for v in want_v)
    print("RESULT " + hdr)
    for name in names:
        b = RES.get((name, base))
        row = f"{name:20s}"
        for v in want_v:
            e = RES.get((name, v))
            row += f"{(b[0] / e[0]):12.3f}" if (e and b) else f"{'--':>12s}"
        print("RESULT " + row)
    print("RESULT ---- pcc ----")
    for name in names:
        row = f"{name:20s}"
        for v in want_v:
            e = RES.get((name, v))
            row += f"{e[2]:12.6f}" if e else f"{'--':>12s}"
        print("RESULT " + row)


main()
