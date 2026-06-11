"""LOG + CHECK_EXIT handlers (Member 2) — REAL, deterministic.

LOG:        append one ledger row from ctx.state["last_decision"], update
            counters + metric.current (on keep), mark the lever tried. -> CHECK_EXIT
CHECK_EXIT: delegate to exit_policy.check_exit(state). -> ROUTE | DONE | STOPPED

Template for the other M2 handlers. Note the idempotent experiment_id so a
resumed LOG never double-writes.
"""

from __future__ import annotations

from .. import exit_policy, states


def log(ctx) -> str:
    d = ctx.state.get("last_decision") or {}
    it = ctx.state.get("iteration", 0)
    lever = ctx.state.get("selected_lever")
    before, after = d.get("before"), d.get("after")
    row = {
        "experiment_id": f"{ctx.run.run_id}#{it}",  # idempotent replay key
        "iteration": it,
        "bucket": ctx.state.get("current_bucket"),
        "lever": lever,
        "result": d.get("result"),
        "reason": d.get("reason"),
        "before": before,
        "after": after,
        "delta": (None if before is None or after is None else round(after - before, 4)),
        "pcc": d.get("pcc"),
        "hypothesis": d.get("hypothesis"),
    }
    ctx.ledger.append(row)

    if lever and lever not in ctx.state.setdefault("tried", []):
        ctx.state["tried"].append(lever)
    if d.get("result") == "keep" and after is not None:
        ctx.state["metric"]["current"] = after
    ctx.state["iteration"] = it + 1
    return states.CHECK_EXIT


def check_exit(ctx) -> str:
    decision = exit_policy.check_exit(ctx.state)  # "continue" | "DONE" | "STOPPED"
    return states.ROUTE if decision == "continue" else decision
