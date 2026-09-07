# SPDX-License-Identifier: Apache-2.0
"""IDEA A bake-off: does the RESIDENT regime search want a SMALLER BLOCK_ROWS at a
given ring depth (more row-blocks per core to pipeline read/compute/write over)?

ISOLATION.  The real op is never touched.  This bench imports the FORKED descriptor
(`desc_fork.py` in this dir, whose `KERNEL_DIR` therefore resolves to this dir's
`kernels/` copy) and monkeypatches it into the op module in place of the shipped
`create_program_descriptor`.  Two consequences, both deliberate:
  * every variant -- INCLUDING the baseline -- runs the same forked kernel PATH, so
    the JIT cache key is identical across variants and the (measured) content-blind
    cache hazard cannot make one variant inherit another's binary;
  * `RMS_RBC=shipped` reproduces the shipped D41 rule verbatim, so `baseline` is the
    op's current approach and not a strawman.

Precision contract is FIXED: each case's own math_fidelity / fp32_dest_acc_en /
dtype come straight out of feature_spec, identical for every variant.

env:
  RMS_RBC / RMS_RBC_MAXB / RMS_RBC_BR / RMS_RBC_DEPTH / RMS_RBC_DEPTHS  -> the rule
  RMS_TAG      label stamped on every RESULT line
  RMS_CASES    comma list of perf-case indices (default: all 19)
  RMS_EXTRA    "1" -> also run the non-perf-group regime sweep (ROW_RESIDENT / STREAM
               / interleaved decode / RM)
  RMS_READS    readings per case (default 1; device kernel time has no warm-up)
"""

import importlib.util
import os
import pathlib
import statistics

for _k, _v in (
    ("TT_METAL_DEVICE_PROFILER", "1"),
    ("TT_METAL_PROFILER_MID_RUN_DUMP", "1"),
    ("TT_METAL_PROFILER_CPP_POST_PROCESS", "1"),
    ("TT_METAL_LOGGER_LEVEL", "error"),
):
    os.environ.setdefault(_k, _v)

import torch  # noqa: E402
import ttnn  # noqa: E402

from eval.sharding import shard_config  # noqa: E402
from eval.golden_tests.rms_norm_ttnn import feature_spec as FS  # noqa: E402
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn as OP  # noqa: E402

# ---- load the fork by PATH (no __init__.py under perf_experiments/ -- ttnn's
# operations/__init__.py walk_packages()-executes anything that becomes a package).
# ABSOLUTE: tt-probe.sh copies this script into tests/.../probes/, so __file__
# would not resolve the fork.  Keep the fork+kernels path pinned to this dir.
_HERE = pathlib.Path(
    "/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/rms_norm_ttnn_run1/tt-metal"
    "/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/resident_block_count"
)
_spec = importlib.util.spec_from_file_location("rbc_desc_fork", _HERE / "desc_fork.py")
_fork = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fork)
OP.create_program_descriptor = _fork.create_program_descriptor

_DUR = "DEVICE KERNEL DURATION [ns]"
TAG = os.environ.get("RMS_TAG", "run")
READS = int(os.environ.get("RMS_READS", "1"))


def _ns(device):
    ttnn.ReadDeviceProfiler(device)
    total, found = 0.0, False
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            e = (getattr(program, "program_analyses_results", None) or {}).get(_DUR)
            if e is None:
                continue
            total += float(e.duration)
            found = True
    return total if found else float("nan")


def pcc(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-30))


def perf_cases():
    return [c for c in FS.LOOSE_CASES if c.get("group") == "perf"]


# ---- the DOMAIN sweep beyond the perf group.  Each entry is a synthetic case in the
# same dict shape, chosen to land in a regime the perf group does not reach:
#   * a wide interleaved prefill that must chunk the width (ROW_RESIDENT / STREAM),
#   * a WIDTH shard (one tile-row per core -- D41's carve-out),
#   * a BLOCK shard whose block count the shallow ring already reaches,
#   * a ROW_MAJOR input (depth ladder collapses to (1,)),
#   * a non-tile-aligned width (PARTIAL_W != 0).
def extra_cases():
    ML = ttnn.TensorMemoryLayout
    base = dict(
        dtype=ttnn.bfloat16,
        fp32_dest_acc_en=False,
        layout=ttnn.TILE_LAYOUT,
        gamma_mode="gamma",
        gamma_dtype=ttnn.bfloat16,
        gamma_layout=ttnn.TILE_LAYOUT,
        group="extra",
        memory_layout=ML.INTERLEAVED,
        extras={"achievable_ns": 1.0, "math_fidelity": ttnn.MathFidelity.HiFi2, "pcc_threshold": 0.9995},
    )

    def mk(shape, **kw):
        c = dict(base)
        c["extras"] = dict(base["extras"])
        c.update(kw)
        c["inputs"] = (shape,)
        return c

    return [
        mk((1, 1, 8192, 16384)),  # very wide prefill -> chunked width
        mk((1, 1, 1024, 16384)),  # wide, fewer rows
        mk((1, 1, 8192, 5120), gamma_mode="gamma_bias_residual", fp32_dest_acc_en=True),
        mk((1, 1, 4096, 1024)),  # 4 tile-rows/core -> br in {1,2,3,4}
        mk((1, 1, 2048, 1024)),  # 2 tile-rows/core
        mk((1, 1, 8192, 1024), dtype=ttnn.float32, fp32_dest_acc_en=True),
        mk((1, 1, 8192, 1000)),  # PARTIAL_W != 0
        mk((1, 1, 128, 4096), layout=ttnn.ROW_MAJOR_LAYOUT, gamma_layout=ttnn.ROW_MAJOR_LAYOUT),
        mk(
            (1, 1, 8192, 1024),
            memory_layout=ML.WIDTH_SHARDED,
            extras={**base["extras"], "shard_shape": [8192, 128], "core_grid": (8, 1)},
        ),
        mk(
            (1, 1, 4096, 1024),
            memory_layout=ML.BLOCK_SHARDED,
            extras={**base["extras"], "shard_shape": [512, 128], "core_grid": (8, 8)},
        ),
    ]


def label(i, c):
    shape = c["inputs"][0]
    ml = str(c["memory_layout"]).split(".")[-1][:5]
    g = c.get("gamma_mode", "gamma")
    f32 = "T" if c.get("fp32_dest_acc_en") else "F"
    gl = "RM" if c.get("gamma_layout") == ttnn.ROW_MAJOR_LAYOUT else "TI"
    lay = "RM" if c.get("layout") == ttnn.ROW_MAJOR_LAYOUT else "TI"
    dt = str(c["dtype"]).split(".")[-1][:7]
    return f"{i:02d} {shape[-2]}x{shape[-1]} {ml} {g[:18]} f32{f32} w{gl} x{lay} {dt}"


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

    expected = OP.torch_rms_norm_ttnn(
        ref["input_tensor"],
        epsilon=1e-12,
        weight=ref.get("weight"),
        bias=ref.get("bias"),
        residual_input_tensor=ref.get("residual_input_tensor"),
    )
    live = [x] + [v for v in kwargs.values() if isinstance(v, ttnn.Tensor)]
    return (lambda: OP.rms_norm_ttnn(x, **kwargs)), expected, live


def main():
    cases = perf_cases()
    if os.environ.get("RMS_EXTRA", "0") not in ("", "0"):
        cases = cases + extra_cases()
    want = os.environ.get("RMS_CASES", "")
    idx = set(int(t) for t in want.split(",") if t.strip()) if want else set(range(len(cases)))

    device = ttnn.open_device(device_id=0)
    try:
        for i, c in enumerate(cases):
            if i not in idx:
                continue
            lbl = label(i, c)
            try:
                run, expected, live = build(device, c)
                out = run()
                p = pcc(ttnn.to_torch(out), expected)
                ttnn.deallocate(out)
                ttnn.synchronize_device(device)
                _ns(device)
                s = []
                for _ in range(READS):
                    o = run()
                    ttnn.synchronize_device(device)
                    v = _ns(device)
                    ttnn.deallocate(o)
                    if v == v:
                        s.append(v)
                m = statistics.median(s) if s else float("nan")
                for t in live:
                    try:
                        ttnn.deallocate(t)
                    except Exception:
                        pass
            except Exception as e:
                print(f"RESULT {TAG} {lbl}  ERROR {type(e).__name__}: {e}", flush=True)
                continue
            ach = float(c["extras"].get("achievable_ns", 1.0))
            print(
                f"RESULT {TAG} {lbl} ns={m:.0f} min={min(s) if s else float('nan'):.0f} "
                f"n={len(s)} pcc={p:.6f} ach={ach:.0f}",
                flush=True,
            )
    finally:
        ttnn.close_device(device)


main()
