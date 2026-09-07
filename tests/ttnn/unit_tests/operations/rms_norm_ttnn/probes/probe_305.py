# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""block_stream_granularity — stream a row-block at SUB-BLOCK granularity.

Runs the op out of THIS DIRECTORY's private clone (descriptor + kernels at their own
path, so the JIT cache key can never alias the shipped build).  A variant is a dict of
module-level knobs applied to the cloned descriptor before the call, so every candidate
is the SAME kernels at the SAME user precision contract (each case's own
math_fidelity / fp32_dest_acc_en / dtypes, straight out of feature_spec) and only the
reader -> compute -> writer HANDOVER GRANULARITY moves.

Cases are read from `eval/golden_tests/rms_norm_ttnn/feature_spec.py` (the `perf`
group, by index) plus a few named extras for regimes the perf group does not reach.

    BSG_VARIANTS=base,br1_d3 BSG_CASES=p04 \
        scripts/tt-probe.sh rms_norm_ttnn < bsg_bench.py

One fresh reading per (variant, case) by default -- device kernel time has no warm-up
transient.  BSG_READS raises it where a small shape's spread needs bounding.
"""

import os
import statistics
import sys

for _k, _v in (
    ("TT_METAL_DEVICE_PROFILER", "1"),
    ("TT_METAL_PROFILER_MID_RUN_DUMP", "1"),
    ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
    ("TT_METAL_LOGGER_LEVEL", "error"),
):
    os.environ.setdefault(_k, _v)

# ABSOLUTE: tt-probe.sh re-homes the script under tests/.../probes/, so __file__ is not here.
HERE = "/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/block_stream_granularity"
sys.path.insert(0, HERE)

import torch  # noqa: E402
import ttnn  # noqa: E402

import rms_norm_ttnn_program_descriptor as PD  # noqa: E402
from rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn  # noqa: E402

from eval.sharding import shard_config  # noqa: E402
from eval.golden_tests.rms_norm_ttnn import feature_spec as FS  # noqa: E402

_DUR = "DEVICE KERNEL DURATION [ns]"
READS = int(os.environ.get("BSG_READS", "1"))


# ---------------------------------------------------------------------------
# variants.  Every knob is inert at its shipped default, so `base` builds the
# shipped program (only its kernel PATH differs, which is a cache key, not a
# program change).
# ---------------------------------------------------------------------------
VARIANTS = {
    "base": {},
    # ---- HOST half: a finer row-block than D41's search can reach ---------
    # D41 picks the candidate with the MOST row-blocks per core but only ever
    # gets a smaller block by buying a DEEPER ring -- it never takes a block
    # FINER than the fit allows at a given depth.  These reach that point.
    "br1": {"BSG_FORCE_BLOCK_ROWS": 1},
    "br1_d3": {"BSG_FORCE_BLOCK_ROWS": 1, "CB_DEPTH_CANDIDATES_RESIDENT": (3,)},
    "br1_d4": {"BSG_FORCE_BLOCK_ROWS": 1, "CB_DEPTH_CANDIDATES_RESIDENT": (4,)},
    "br1_d6": {"BSG_FORCE_BLOCK_ROWS": 1, "CB_DEPTH_CANDIDATES_RESIDENT": (6,)},
    "br1_d8": {"BSG_FORCE_BLOCK_ROWS": 1, "CB_DEPTH_CANDIDATES_RESIDENT": (8,)},
    "br2_d4": {"BSG_FORCE_BLOCK_ROWS": 2, "CB_DEPTH_CANDIDATES_RESIDENT": (4,)},
    "d2": {"CB_DEPTH_CANDIDATES_RESIDENT": (2,)},  # pre-D41 reference point
    # ---- KERNEL half: sub-block handover INSIDE the block ----------------
    "pa_row": {"BSG_MODE": 1},
    "pb_row": {"BSG_MODE": 2},
    "pa_pb_row": {"BSG_MODE": 3},
    # ---- reader sub-tile-row pushes --------------------------------------
    "rd2": {"BSG_READ_SPLIT": 2},
    "rd2_pa": {"BSG_READ_SPLIT": 2, "BSG_MODE": 1},
    "rd4_pa": {"BSG_READ_SPLIT": 4, "BSG_MODE": 1},
}

# ---------------------------------------------------------------------------
# extra cases (regimes the perf group does not carry).  Same builder.
# ---------------------------------------------------------------------------
EXTRA = {
    # ROW_MAJOR activation, TILE weight -- a guard-set cell
    "rm1024": {
        "inputs": [(1, 1, 8192, 1024)],
        "dtype": ttnn.bfloat16,
        "layout": ttnn.ROW_MAJOR_LAYOUT,
        "memory_layout": ttnn.TensorMemoryLayout.INTERLEAVED,
        "gamma_mode": "gamma",
        "extras": {},
    },
    # HEIGHT shard, native in/out, no combine -- reader does NO NoC read
    "height2048": {
        "inputs": [(1, 1, 2048, 256)],
        "dtype": ttnn.bfloat16,
        "layout": ttnn.TILE_LAYOUT,
        "memory_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "gamma_mode": "gamma",
        "extras": {"shard_shape": (32, 256), "core_grid": (8, 8)},
    },
    # fewer tile-rows per core than the focus shape (1 tile-row/core exactly)
    "h3520": {
        "inputs": [(1, 1, 3520, 1024)],
        "dtype": ttnn.bfloat16,
        "layout": ttnn.TILE_LAYOUT,
        "memory_layout": ttnn.TensorMemoryLayout.INTERLEAVED,
        "gamma_mode": "gamma",
        "extras": {},
    },
    # 4 and 8 tile-rows per core
    "h14080": {
        "inputs": [(1, 1, 14080, 1024)],
        "dtype": ttnn.bfloat16,
        "layout": ttnn.TILE_LAYOUT,
        "memory_layout": ttnn.TensorMemoryLayout.INTERLEAVED,
        "gamma_mode": "gamma",
        "extras": {},
    },
    "h28160": {
        "inputs": [(1, 1, 28160, 1024)],
        "dtype": ttnn.bfloat16,
        "layout": ttnn.TILE_LAYOUT,
        "memory_layout": ttnn.TensorMemoryLayout.INTERLEAVED,
        "gamma_mode": "gamma",
        "extras": {},
    },
}


def _perf_cases():
    return [c for c in FS.LOOSE_CASES if c.get("group") == "perf"]


def _all_cases():
    d = {}
    for i, c in enumerate(_perf_cases()):
        d[f"p{i:02d}"] = c
    d.update(EXTRA)
    return d


def _label(c):
    shape = c["inputs"][0]
    ml = str(c["memory_layout"]).split(".")[-1][:5]
    g = c.get("gamma_mode", "gamma")
    f32 = "T" if c.get("fp32_dest_acc_en") else "F"
    lay = "RM" if c["layout"] == ttnn.ROW_MAJOR_LAYOUT else "TI"
    return f"{shape[-2]}x{shape[-1]}/{ml}/{g[:9]}/f32{f32}/{lay}"


def _pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def _read_ns(dev):
    ttnn.ReadDeviceProfiler(dev)
    total, found = 0.0, False
    for progs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for p in progs:
            e = (getattr(p, "program_analyses_results", None) or {}).get(_DUR)
            if e is not None:
                total += float(e.duration)
                found = True
    return total if found else float("nan")


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


def main():
    cases = _all_cases()
    want_v = os.environ.get("BSG_VARIANTS", "base").split(",")
    want_c = os.environ.get("BSG_CASES", "p04").split(",")

    dev = ttnn.open_device(device_id=0)
    try:
        for cname in want_c:
            c = cases[cname]
            lbl = _label(c)
            try:
                run, expected, live = build(dev, c)
            except Exception as e:
                print(f"BSG {cname} {lbl} BUILD-FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            for vname in want_v:
                knobs = VARIANTS[vname]
                saved = {k: getattr(PD, k) for k in knobs}
                for k, v in knobs.items():
                    setattr(PD, k, v)
                try:
                    out = run()
                    p = _pcc(ttnn.to_torch(out), expected)
                    del out
                    ttnn.synchronize_device(dev)
                    _read_ns(dev)
                    s = []
                    for _ in range(READS):
                        run()
                        ttnn.synchronize_device(dev)
                        v = _read_ns(dev)
                        if v == v:
                            s.append(v)
                    m = statistics.median(s) if s else float("nan")
                    print(
                        f"BSG {cname:9s} {vname:10s} ns={m:10.0f} min={min(s):10.0f} n={len(s)} "
                        f"pcc={p:.6f} {lbl}",
                        flush=True,
                    )
                except Exception as e:  # a variant that cannot build is DATA, not a crash
                    print(f"BSG {cname:9s} {vname:10s} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
                finally:
                    for k, v in saved.items():
                        setattr(PD, k, v)
            for t in live:
                try:
                    ttnn.deallocate(t)
                except Exception:
                    pass
    finally:
        ttnn.close_device(dev)


main()
