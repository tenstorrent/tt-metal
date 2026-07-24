"""Agent Loop entry point — continues from a finished Before Loop (PLAN 8).

    python -m agent.loop [runs_root] [--until STATE]

Reads runs/latest/state.json (must be at BEFORE_LOOP_DONE), then drives the
state machine. `--until ROUTE` runs through ROUTE and stops (no key needed —
SELECT/APPLY never run), so you can inspect runs/latest/route_briefs.jsonl.
Real SELECT/REPAIR handlers load .env.agent themselves (section 3.1).
"""

from __future__ import annotations

import argparse

from . import engine, states
from .handlers import build_handlers
from .loop_context import LoopContext
from .looplog import make_logger

_STATES = sorted(states.TRANSITIONS)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="agent.loop")
    ap.add_argument("runs_root", nargs="?", default="runs")
    ap.add_argument("--until", choices=_STATES, help="run through this stage then stop (e.g. ROUTE)")
    args = ap.parse_args(argv)

    ctx = LoopContext.from_latest(args.runs_root)
    if ctx.state["state"] in states.TERMINAL:
        print(f"run already terminal: {ctx.state['state']}")
        return 0

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
