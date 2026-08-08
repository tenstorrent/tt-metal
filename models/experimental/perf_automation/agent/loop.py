"""Agent Loop entry point — continues from a finished Before Loop (PLAN 8).

    python -m agent.loop [runs_root]

Reads runs/latest/state.json (must be at BEFORE_LOOP_DONE), then drives the
state machine to a terminal. Real SELECT/REPAIR handlers load .env.agent
themselves (§3.1); the mock skeleton needs no credentials.
"""

from __future__ import annotations

import sys

from . import engine, states
from .handlers import build_handlers
from .loop_context import LoopContext


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    runs_root = argv[0] if argv else "runs"

    ctx = LoopContext.from_latest(runs_root)
    if ctx.state["state"] in states.TERMINAL:
        print(f"run already terminal: {ctx.state['state']}")
        return 0

    final = engine.run(ctx, build_handlers())
    m = ctx.state.get("metric", {})
    print(
        f"loop finished: {final}  (iteration {ctx.state.get('iteration')}, "
        f"{m.get('name')} {m.get('current')} / target {m.get('target')})"
    )
    return 0 if final in (states.DONE, states.STOPPED) else 1


if __name__ == "__main__":
    raise SystemExit(main())
