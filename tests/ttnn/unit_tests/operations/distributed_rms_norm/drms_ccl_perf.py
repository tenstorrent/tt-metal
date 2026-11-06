"""Device time of a multi-chip (fused-CCL) op, measured the same way every time.

Protocol (validated on qb2-120, 1x4 ring, distributed_rms_norm 640x1792: one experiment of 500 calls repeats to
+-0.05 us across 100 experiments):

  pattern     a spin op (one core waits ~1 ms, next to nothing to dispatch) then `iters` op calls, one trace
  experiment  the trace executed `patterns` times back to back, one profiler read: every chip's DEVICE KERNEL
              DURATION of patterns * iters calls (the spins are dropped)
  run         `experiments` experiments: CHECK_EXPERIMENTS while iterating, FINAL_EXPERIMENTS for the final numbers

The spin makes every chip start the calls together: it hides the op's dispatch and leaves a small launch skew. It
hides dispatch completely, so a separate check runs the op back to back from one trace and reports the per-chip gap
between one call's end and the next one's launch: above DISPATCH_BOUND_GAP_US the op is dispatch-bound (the
dispatcher, not the device, sets its rate), which the device time cannot show.

Metrics, each the mean over an experiment's calls, then mean / stdev over experiments:
  fastest   the chip that finished soonest in each call: the op's own work with the least waiting on its peers.
            THE OPTIMIZATION TARGET.
  slowest   the chip that finished last in each call: when the call is done on every chip (the per-call cost a
            model pays). Recorded and guarded: a change must not regress it.
  average   the mean over chips; spread = slowest - fastest per call (the cost of the chips not being in step).
  per_chip  each chip's own mean over all calls (is one chip inherently slower).

compare(base, cand) accepts a change when `fastest` improves beyond the noise and `slowest` does not regress beyond
it: the fastest chip alone would also improve when a change merely makes the chips more skewed.

Needs the device profiler (PROFILER_ENV) set before the mesh opens, and a trace region. Per-op entry points live in
the golden suites (eval/golden_tests/<op>/perf_measure.py). Host side:

    python -m eval.ccl_perf show <result.json>
    python -m eval.ccl_perf compare <base.json> <candidate.json>
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from pathlib import Path

DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
SPIN_KERNEL = Path(__file__).with_name("ccl_perf_spin.cpp")
SPIN_CYCLES = 1_350_000  # ~1 ms at Blackhole's 1350 MHz AICLK: only needs to outlast the dispatch of `iters` calls
CHECK_EXPERIMENTS = 5
FINAL_EXPERIMENTS = 100
PATTERNS = 100
ITERS = 5
PROFILER_PROGRAMS = 1000  # programs the device profiler holds between reads (DEFAULT_PROFILER_PROGRAM_SUPPORT_COUNT)
DISPATCH_CALLS = 20
DISPATCH_BOUND_GAP_US = 1.0  # back to back, a call launches ~0.6 us after the previous one ends unless dispatch-bound
TRACE_REGION_SIZE = 1 << 25
# Set before the mesh opens. The CSV log is not needed (durations come from get_latest_programs_perf_data) and
# would grow by ~100 MB per experiment.
PROFILER_ENV = {
    "TT_METAL_DEVICE_PROFILER": "1",
    "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
    "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
    "TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES": "1",
}
METRICS = ("fastest", "slowest", "average", "spread")
_PER_CALL = {
    "fastest": min,
    "slowest": max,
    "average": st.mean,
    "spread": lambda c: max(c) - min(c),
}


# ---- device side ----


def chips_of(mesh) -> list[int]:
    """Device ids in mesh row-major order (the column order of every [call][chip] row)."""
    import ttnn

    rows, cols = tuple(mesh.shape)
    return [int(mesh.get_device_id(ttnn.MeshCoordinate(r, c))) for r in range(rows) for c in range(cols)]


def spin_op(mesh, cycles: int = SPIN_CYCLES):
    """A zero-arg callable dispatching the spin program on every chip (one core, compile-time cycle count)."""
    import ttnn

    core = ttnn.CoreCoord(0, 0)
    program = ttnn.ProgramDescriptor()
    program.kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(SPIN_KERNEL),
            core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
            compile_time_args=[int(cycles)],
            runtime_args=ttnn.RuntimeArgs(),
            config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1),
        )
    ]
    rows, cols = tuple(mesh.shape)
    desc = ttnn.MeshProgramDescriptor()
    desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(0, 0), ttnn.MeshCoordinate(rows - 1, cols - 1))] = program
    io = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 32]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    return lambda: ttnn.generic_op([io, io], desc)


def _read(mesh, chips) -> dict:
    """{chip: [(start cycles, end cycles, duration ns)]} of the programs since the last read, in launch order."""
    import ttnn

    ttnn.ReadDeviceProfiler(mesh)
    out = {c: [] for c in chips}
    for dev, programs in (ttnn.get_latest_programs_perf_data() or {}).items():
        rs = [
            p.program_analyses_results[DURATION_KEY]
            for p in programs
            if DURATION_KEY in (p.program_analyses_results or {})
        ]
        out[int(dev)] = sorted(((r.start_timestamp, r.end_timestamp, float(r.duration)) for r in rs))
    return out


def measure(
    mesh,
    op,
    *,
    experiments: int,
    patterns: int = PATTERNS,
    iters: int = ITERS,
    spin_cycles: int = SPIN_CYCLES,
    log=print,
) -> dict:
    """Run the protocol. `op` is a zero-arg callable dispatching one call on every chip (warm-up included here).
    Returns {"chips", "ns": [experiment][call][chip], "spin_us"}."""
    import ttnn

    assert patterns * (iters + 1) <= PROFILER_PROGRAMS, "one experiment must fit the profiler's program buffer"
    chips = chips_of(mesh)
    spin = spin_op(mesh, spin_cycles)
    spin()
    op()
    ttnn.synchronize_device(mesh)
    _read(mesh, chips)  # drop the warm-up
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    spin()
    for _ in range(iters):
        op()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    runs, spins = [], []
    try:
        for e in range(experiments):
            for _ in range(patterns):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            per = _read(mesh, chips)
            want = patterns * (iters + 1)
            got = {c: len(per[c]) for c in chips}
            assert all(n == want for n in got.values()), f"programs per chip {got}, expected {want}"
            calls = [[per[c][i][2] for c in chips] for i in range(want) if i % (iters + 1)]
            spins += [per[chips[0]][i][2] for i in range(0, want, iters + 1)]
            runs.append(calls)
            if log and (e + 1) % max(1, experiments // 10) == 0:
                log(f"CCL_PERF experiment {e + 1}/{experiments}")
    finally:
        ttnn.release_trace(mesh, tid)
    return {"chips": chips, "ns": runs, "spin_us": st.median(spins) / 1e3}


def dispatch_check(mesh, op, *, calls: int = DISPATCH_CALLS) -> dict:
    """`calls` op calls back to back from one trace: per chip, the median gap from a call's end to the next launch,
    the median launch-to-launch period and the median device time (one chip's own clock, no cross-chip sync)."""
    import ttnn

    chips = chips_of(mesh)
    op()
    ttnn.synchronize_device(mesh)
    _read(mesh, chips)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    for _ in range(calls):
        op()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    ttnn.release_trace(mesh, tid)
    per = _read(mesh, chips)
    out = {"calls": calls, "per_chip": {}}
    for c in chips:
        ps = per[c]
        assert len(ps) == calls, f"chip {c}: {len(ps)} programs, expected {calls}"
        ns_per_cycle = st.median(d / (e - s) for s, e, d in ps if e > s)
        gap = [(ps[i + 1][0] - ps[i][1]) * ns_per_cycle / 1e3 for i in range(calls - 1)]
        period = [(ps[i + 1][0] - ps[i][0]) * ns_per_cycle / 1e3 for i in range(calls - 1)]
        out["per_chip"][str(c)] = {
            "gap_us": st.median(gap),
            "period_us": st.median(period),
            "device_us": st.median(d for _, _, d in ps) / 1e3,
        }
    out["max_gap_us"] = max(v["gap_us"] for v in out["per_chip"].values())
    out["max_period_us"] = max(v["period_us"] for v in out["per_chip"].values())
    out["dispatch_bound"] = out["max_gap_us"] > DISPATCH_BOUND_GAP_US
    return out


# ---- host side ----


def _stats(v: list[float], per_call: list[float]) -> dict:
    """mean / stdev / min / max over experiment means; sem = the noise of the mean (from the experiment spread when
    there are >= 2 experiments, else from the per-call spread)."""
    n = len(v)
    sd = st.stdev(v) if n > 1 else 0.0
    if n > 1:
        sem = sd / math.sqrt(n)
    else:
        sem = st.stdev(per_call) / math.sqrt(len(per_call)) if len(per_call) > 1 else 0.0
    return {"mean": st.mean(v), "stdev": sd, "min": min(v), "max": max(v), "sem": sem}


def summarize(result: dict) -> dict:
    ns, chips = result["ns"], result["chips"]
    calls = [c for r in ns for c in r]
    out = {"experiments": len(ns), "calls": len(calls), "target": "fastest"}
    for m in METRICS:
        f = _PER_CALL[m]
        out[m] = _stats([st.mean(f(c) for c in r) / 1e3 for r in ns], [f(c) / 1e3 for c in calls])
    out["per_chip"] = {
        str(chip): _stats([st.mean(c[p] for c in r) / 1e3 for r in ns], [c[p] / 1e3 for c in calls])
        for p, chip in enumerate(chips)
    }
    last = [max(range(len(chips)), key=lambda p: c[p]) for c in calls]
    out["slowest_share"] = {str(chip): last.count(p) / len(calls) for p, chip in enumerate(chips)}
    return out


def compare(base: dict, cand: dict, *, k: float = 3.0, min_us: float = 0.05) -> dict:
    """Candidate vs baseline summaries: a win needs `fastest` lower by more than the noise (k combined sems, at least
    min_us) with `slowest` not higher by more than the same margin."""
    out = {}
    for m in ("fastest", "slowest"):
        d = cand[m]["mean"] - base[m]["mean"]
        thr = max(k * math.hypot(base[m]["sem"], cand[m]["sem"]), min_us)
        out[m] = {"delta_us": d, "noise_us": thr}
    if out["slowest"]["delta_us"] > out["slowest"]["noise_us"]:
        out["verdict"] = "reject: slowest chip regressed"
    elif out["fastest"]["delta_us"] < -out["fastest"]["noise_us"]:
        out["verdict"] = "win"
    elif out["fastest"]["delta_us"] > out["fastest"]["noise_us"]:
        out["verdict"] = "regression"
    else:
        out["verdict"] = "null (within noise)"
    return out


def format_summary(s: dict, dispatch: dict | None = None, label: str = "") -> str:
    f = lambda x: f"{x['mean']:.3f} us (stdev {x['stdev']:.3f}, +-{x['sem']:.3f})"  # noqa: E731
    lines = [f"CCL_PERF {label}: {s['experiments']} experiments, {s['calls']} calls"]
    lines.append(f"  fastest chip  {f(s['fastest'])}   <- optimization target")
    lines.append(f"  slowest chip  {f(s['slowest'])}")
    lines.append(f"  chip average  {f(s['average'])}")
    lines.append(f"  spread        {f(s['spread'])}   (slowest - fastest per call)")
    lines.append(
        "  per chip      "
        + "  ".join(f"dev{c} {v['mean']:.3f}" for c, v in s["per_chip"].items())
        + "   slowest share "
        + " ".join(f"dev{c} {100 * v:.0f}%" for c, v in s["slowest_share"].items())
    )
    if dispatch is not None:
        tag = "DISPATCH-BOUND" if dispatch["dispatch_bound"] else "not dispatch-bound"
        lines.append(
            f"  dispatch      {tag}: back to back, max gap {dispatch['max_gap_us']:.2f} us, "
            f"max period {dispatch['max_period_us']:.2f} us per call"
        )
    return "\n".join(lines)


def run(
    mesh, op, *, experiments: int, out_path: Path | None = None, label: str = "", meta: dict | None = None, log=print
) -> dict:
    """measure + dispatch_check + summarize; writes {meta, chips, ns, summary, dispatch} to out_path (JSON)."""
    result = measure(mesh, op, experiments=experiments, log=log)
    dispatch = dispatch_check(mesh, op)
    summary = summarize(result)
    doc = {
        "meta": {**(meta or {}), "label": label, "patterns": PATTERNS, "iters": ITERS, "spin_us": result["spin_us"]},
        "chips": result["chips"],
        "ns": result["ns"],
        "summary": summary,
        "dispatch": dispatch,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(doc))
    if log:
        log(format_summary(summary, dispatch, label))
    return doc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    p_show = sub.add_parser("show", help="print a result's summary")
    p_show.add_argument("result", type=Path)
    p_cmp = sub.add_parser("compare", help="candidate vs baseline: win / null / regression / reject")
    p_cmp.add_argument("base", type=Path)
    p_cmp.add_argument("cand", type=Path)
    a = p.parse_args(argv)
    if a.cmd == "show":
        doc = json.loads(a.result.read_text())
        print(format_summary(doc["summary"], doc.get("dispatch"), doc["meta"].get("label", "")))
        return 0
    base, cand = (json.loads(x.read_text()) for x in (a.base, a.cand))
    c = compare(base["summary"], cand["summary"])
    for m in ("fastest", "slowest"):
        print(
            f"{m:8s} {base['summary'][m]['mean']:.3f} -> {cand['summary'][m]['mean']:.3f} us  "
            f"delta {c[m]['delta_us']:+.3f} (noise {c[m]['noise_us']:.3f})"
        )
    print(f"verdict: {c['verdict']}")
    for doc, name in ((base, "base"), (cand, "candidate")):
        if doc.get("dispatch", {}).get("dispatch_bound"):
            print(f"WARNING: {name} is dispatch-bound (max gap {doc['dispatch']['max_gap_us']:.2f} us)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
