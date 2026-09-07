# Perf experiment `per_channel_mcast_v2` -- the variant driver.
#
#   head   the SHIPPED op, untouched (ttnn/.../kernels)      <- the graduation baseline
#   base   the shipped descriptor with KERNEL_DIR = k_base   <- proves the kernel COPY is neutral
#   off    pd_mcast with the multicast OFF + k_mcast         <- the ENTRY CONDITION check
#   col    Mcast1D PerColumn  (round 1's geometry)
#   row    Mcast1D PerRow
#   one    Mcast2D, ONE injector -> the whole rectangle
#   splitN Mcast2D rotating: N injectors, each reads 72/N tiles and broadcasts its slice
#
# One measured run per (case, variant); RMS_REPS repeats the whole matrix.
import importlib
import importlib.util
import os
import sys
from pathlib import Path

# tt-probe.sh re-homes the script into the probes dir, so HERE is pinned, not derived.
HERE = Path(
    os.environ.get(
        "RMS_EXP_DIR",
        "/localdev/dnijemcevic/2026_09_04/1519_dnijemcevic_agent_eval_new/clones/"
        "rms_norm_ttnn_run1/tt-metal/ttnn/ttnn/operations/rms_norm_ttnn/perf_experiments/per_channel_mcast_v2",
    )
)
sys.path.insert(0, str(HERE))

import bench_v2  # noqa: E402  (sets the profiler env before ttnn is imported)
import ttnn  # noqa: E402
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD  # noqa: E402

OP = importlib.import_module("ttnn.operations.rms_norm_ttnn.rms_norm_ttnn")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PDM = _load("pd_mcast_v2", HERE / "pd_mcast.py")
SHIPPED = OP.create_program_descriptor
SHIPPED_KDIR = PD.KERNEL_DIR

# ===========================================================================
# THE JIT-BUILD HAZARD THAT SANK ROUND 1's OFF PATH -- and the one line that fixes it.
# ===========================================================================
# `ttnn/ttnn/operations/__init__.py` walks EVERY package under `operations/` with
# `pkgutil.walk_packages` and executes it.  Two sibling perf-experiment dirs carry an
# `__init__.py` AND a module that does `os.environ["RMS_STAGE_ZONES"] = "1"` at import
# (fold_gather_overlap/test_zones_fold_overlap.py, reader_boot_order/zones_boot_order.py),
# so `import ttnn` FLIPS THE PROFILING DEFINE ON, process-wide, part-way through its own
# walk.  The shipped descriptor module is imported EARLIER in that walk and reads False;
# any experiment descriptor loaded afterwards reads True.
#
# The consequence is exactly round 1's symptom: the candidate compiles the
# zone-instrumented reader/writer/compute and the baseline compiles the clean ones, so
# every non-engaged plan measures ~4-5% slower for a reason that has nothing to do with
# the multicast.  Traced with an `os.environ.__setitem__` hook, not guessed.
#
# Reconcile explicitly -- the experiment's descriptor uses the SHIPPED module's value.
if PDM.STAGE_ZONES != PD.STAGE_ZONES:
    print(
        f"BUILD reconciling STAGE_ZONES: experiment={PDM.STAGE_ZONES} -> shipped={PD.STAGE_ZONES} "
        f"(env={os.environ.get('RMS_STAGE_ZONES')!r}, flipped by a sibling experiment's import)",
        flush=True,
    )
    PDM.STAGE_ZONES = PD.STAGE_ZONES

print(f"BUILD stage_zones={PD.STAGE_ZONES} (shipped and experiment agree)", flush=True)


def _shipped(kdir):
    def go():
        OP.create_program_descriptor = SHIPPED
        PD.KERNEL_DIR = kdir

    return go


def _exp(**knobs):
    def go():
        OP.create_program_descriptor = PDM.create_program_descriptor
        PDM.KERNEL_DIR = HERE / "k_mcast"
        PDM.PC_MCAST_MODE = knobs.get("mode")
        PDM.PC_MCAST_HANDSHAKE = knobs.get("handshake", False)
        PDM.PC_MCAST_SPLIT = knobs.get("split", 1)
        PDM.PC_MCAST_INJ_TRIM = knobs.get("inj_trim", None)
        PDM.PC_MCAST_STREAM = knobs.get("stream", False)

    return go


VARIANTS = {
    "head": _shipped(SHIPPED_KDIR),
    "base": _shipped(HERE / "k_base"),
    "off": _exp(mode=None),
    "col": _exp(mode="col"),
    "colw": _exp(mode="col", inj_trim=0),
    "row": _exp(mode="row"),
    "one": _exp(mode="one"),
    "onew": _exp(mode="one", inj_trim=0),
    "split4": _exp(mode="split", split=4),
    "split8": _exp(mode="split", split=8),
    "split11": _exp(mode="split", split=11),
    "split22": _exp(mode="split", split=22),
    "split11w": _exp(mode="split", split=11, inj_trim=0),
    "split8s": _exp(mode="split", split=8, stream=True),
    "split11s": _exp(mode="split", split=11, stream=True),
    "ones": _exp(mode="one", stream=True),
}

DEFAULT_CASES = ["FOCUS", "F1024", "FGB", "STREAM", "WSHARD", "BLOCK", "RMW"]


def main():
    cases = (os.environ.get("RMS_CASES") or ",".join(DEFAULT_CASES)).split(",")
    labels = (os.environ.get("RMS_VARIANTS") or "head,off").split(",")
    reps = int(os.environ.get("RMS_REPS", "1"))
    res = {}
    device = ttnn.open_device(device_id=0)
    try:
        for _ in range(reps):
            for label in labels:
                VARIANTS[label]()
                for case in cases:
                    try:
                        ns, p, r = bench_v2.measure(device, case)
                    except Exception as exc:  # a variant that cannot express a case
                        print(f"RESULT.raw {case:8s} {label:9s} FAILED {type(exc).__name__}: {exc}", flush=True)
                        res.setdefault((case, label), []).append((float("nan"), float("nan"), float("nan")))
                        continue
                    res.setdefault((case, label), []).append((ns, p, r))
                    print(f"RESULT.raw {case:8s} {label:9s} ns={ns:10.0f} pcc={p:.6f} relrms={r:.3e}", flush=True)
    finally:
        _shipped(SHIPPED_KDIR)()
        ttnn.close_device(device)

    def best(case, label):
        vals = [x[0] for x in res.get((case, label), []) if x[0] == x[0]]
        return min(vals) if vals else float("nan")

    print("RESULT " + f"{'case':8s}" + "".join(f"{l:>11s}" for l in labels))
    for case in cases:
        print("RESULT " + f"{case:8s}" + "".join(f"{best(case,l):11.0f}" for l in labels))
    print("RESULT --- speedup vs " + labels[0] + " (>1 = faster) ---")
    for case in cases:
        b = best(case, labels[0])
        row = f"{case:8s}" + "".join(f"{b/best(case,l):11.3f}" for l in labels)
        pcs = " ".join(f"{l}:{min([x[1] for x in res.get((case,l),[(0,float('nan'),0)])]):.6f}" for l in labels)
        print("RESULT " + row + "  pcc " + pcs)


main()
