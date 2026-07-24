"""Agent Loop entry point — continues from a finished Before Loop (PLAN 8).

    python -m agent.loop [runs_root] [--until STATE]

Reads runs/latest/state.json (must be at BEFORE_LOOP_DONE), then drives the
state machine. `--until ROUTE` runs through ROUTE and stops (no key needed —
SELECT/APPLY never run), so you can inspect runs/latest/route_briefs.jsonl.
Real SELECT/REPAIR handlers load .env.agent themselves (section 3.1).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import engine, states
from .handlers import build_handlers
from .loop_context import LoopContext
from .looplog import make_logger

_STATES = sorted(states.TRANSITIONS)


def _matmul_sweep_prepass(ctx, args) -> None:
    """Opt-in pre-pass: before the optimize loop starts, run the SEPARATE matmul_sweep module on the
    resolved perf-test node (from runs/latest) and write matmul_sweep.json into the run dir. This only
    ORCHESTRATES the standalone sweep (imports + calls it); it does not change the loop/state machine.
    Idempotent per run (skips if the table already exists) and failure-swallowing so a sweep issue
    never blocks the optimize loop."""
    ptr = (ctx.manifest.get("perf_test_resolved") or {}) if getattr(ctx, "manifest", None) else {}
    node = ptr.get("path")
    if not node:
        print("  [matmul-sweep] no resolved perf-test node in runs/latest; skipping")
        return
    case = ptr.get("case") or None
    out = Path(ctx.run.dir) / "matmul_sweep.json"
    if out.exists():
        print(f"  [matmul-sweep] {out} already present; skipping (delete it to re-sweep)")
        return
    print(f"  [matmul-sweep] pre-pass on {node}{' -k ' + case if case else ''} -> {out}")
    pkg_root = Path(__file__).resolve().parents[1]
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))
    try:
        from cc_optimize.matmul_sweep import run_prepass

        s = run_prepass(
            node,
            case=case,
            out_path=str(out),
            pcc_threshold=args.matmul_sweep_pcc,
            iters=args.matmul_sweep_iters,
            max_shapes=args.matmul_sweep_max_shapes,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  [matmul-sweep] failed ({type(exc).__name__}: {str(exc)[-300:]}); loop continues")
        return
    print(
        f"  [matmul-sweep] {s.get('shapes', 0)} matmul shapes, {s.get('seeded', 0)} seeded, "
        f"{s.get('improved', 0)} beat full-precision -> {out}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agent.loop")
    ap.add_argument("runs_root", nargs="?", default="runs")
    ap.add_argument("--until", choices=_STATES, help="run through this stage then stop (e.g. ROUTE)")
    ap.add_argument(
        "--matmul-sweep",
        action="store_true",
        dest="matmul_sweep",
        help="run the standalone matmul fidelity x dtype sweep pre-pass FIRST (writes matmul_sweep.json "
        "in runs/latest) before the optimize loop begins. OFF by default; when off the loop is unchanged. "
        "The sweep is a separate module -- it does not alter the loop/state machine.",
    )
    ap.add_argument("--matmul-sweep-pcc", type=float, default=0.99, help="matmul-sweep min PCC to accept a config")
    ap.add_argument("--matmul-sweep-iters", type=int, default=5, help="matmul-sweep timed reps per config")
    ap.add_argument("--matmul-sweep-max-shapes", type=int, default=0, help="matmul-sweep distinct-shape cap (0=all)")
    args = ap.parse_args(argv)

    ctx = LoopContext.from_latest(args.runs_root)
    if ctx.state["state"] in states.TERMINAL:
        print(f"run already terminal: {ctx.state['state']}")
        return 0

    if getattr(args, "matmul_sweep", False):
        _matmul_sweep_prepass(ctx, args)

    stop_after = {args.until} if args.until else None
    reached = engine.run(ctx, build_handlers(), stop_after=stop_after, log=make_logger())

    m = ctx.state.get("metric", {})
    if args.until and reached not in states.TERMINAL:
        print(f"ran through {args.until}; parked at {reached}.")
        if ctx.state.get("route_brief_id"):
            print(f"brief_id: {ctx.state['route_brief_id']}  (runs/latest/route_briefs.jsonl)")
    else:
        print(
            f"loop finished: {reached}  (iteration {ctx.state.get('iteration')}, "
            f"{m.get('name')} {m.get('current')} / target {m.get('target')})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
