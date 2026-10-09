"""Replay exploration policies on recorded rounds ("dreaming").

For each policy, beta and recorded round, the policy starts seeing only the round root. Each batch
reveals the recorded child of every selected start point (root -> earliest unopened recorded branch,
head -> that branch's next recorded attempt). Nothing is generated or run. One replay scores

    V = best valid score revealed (or the round root's) - cost * attempts revealed + bonus * attempts / steps

and a policy's objective is V averaged over rounds at its default beta.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from .policy_api import ROOT, PlanContext, RoundView, load_policy, replay_value, validate
from .tree import RecordedRound, round_summary


def policy_default_beta(path: Path, fallback: float) -> float:
    ns: dict = {}
    for line in path.read_text().splitlines():
        if line.startswith("DEFAULT_BETA"):
            exec(line, ns)  # a single literal assignment
            return float(ns["DEFAULT_BETA"])
    return fallback


def simulate(policy_path, beta, rr: RecordedRound, history, defaults, noise_pct, cost, bonus, max_steps):
    pol = load_policy(policy_path, {"beta": beta, "defaults": defaults})
    plan = pol.plan(PlanContext(rr.round, defaults, history))
    view = RoundView(rr.round, plan.W, plan.R, noise_pct, {}, {}, 0, rr.root_score)
    unopened = sorted(rr.branches)
    total = len(rr.nodes)
    trace = {"round": rr.round, "beta": beta, "plan": {"W": plan.W, "R": plan.R, "reason": plan.reason}, "steps": []}
    error = None
    while view.steps_done < max_steps:
        batch = pol.select_batch(view)
        errs = validate(view, batch)
        if errs:
            error = f"step {view.steps_done + 1}: illegal batch: {errs}"
            break
        view.closed.update(batch.closed)
        if not batch.items:
            trace["steps"].append({"step": view.steps_done + 1, "stop": True, "closed": batch.closed})
            break
        revealed = []
        for it in batch.items:
            if it.parent == ROOT:
                if unopened:
                    b = unopened.pop(0)
                    view.branches[b] = [rr.branches[b][0]]
                    revealed.append(rr.branches[b][0])
            else:
                b = next(b for b, ns in view.branches.items() if ns[-1].node_id == it.parent)
                k = len(view.branches[b])
                if k < len(rr.branches[b]):
                    view.branches[b].append(rr.branches[b][k])
                    revealed.append(rr.branches[b][k])
        if not revealed:  # every selected start point is beyond the recording ("out of support")
            trace["steps"].append(
                {"step": view.steps_done + 1, "out_of_support": True, "batch": [i.parent for i in batch.items]}
            )
            break
        view.steps_done += 1
        trace["steps"].append(
            {
                "step": view.steps_done,
                "batch": [{"parent": i.parent, "role": i.role, "why": i.why} for i in batch.items],
                "closed": batch.closed,
                "revealed": [{"node": n.node_id, "score": n.score, "fail_class": n.fail_class} for n in revealed],
            }
        )
        if view.attempts == total:
            break
    n = view.attempts
    best = view.best()
    V = 0.0 if error else replay_value(best, n, view.steps_done, cost, bonus)
    res = {"round": rr.round, "beta": beta, "V": round(V, 4), "attempts": n, "steps": view.steps_done, "best": best}
    if error:
        res["error"] = error
    trace.update({k: v for k, v in res.items() if k != "steps"}, n_steps=res["steps"])  # keep the per-step list
    return res, trace


def replay_policy(
    policy: Path,
    rounds: list[RecordedRound],
    noise: float,
    defaults: dict,
    dcfg: dict,
    sweep: list[float] | None = None,
    traces: list | None = None,
) -> dict:
    """Replay one policy on `rounds` over a beta sweep. Returns the replay.json payload."""
    cost = float(dcfg.get("cost_per_attempt", 0.005))
    bonus = float(dcfg.get("parallel_bonus", 0.01))
    max_steps = int(dcfg.get("max_replay_steps", 100))
    sweep = sweep if sweep is not None else [float(x) for x in dcfg.get("beta_sweep", [0.6])]
    dbeta = policy_default_beta(Path(policy), float(defaults.get("beta", 0.6)))
    betas = sorted(set(sweep) | {dbeta})
    per = []
    for beta in betas:
        for i, rr in enumerate(rounds):
            history = [round_summary(r) for r in rounds[:i]]
            res, tr = simulate(policy, beta, rr, history, defaults, noise, cost, bonus, max_steps)
            per.append(res)
            if traces is not None:
                traces.append({"policy": str(policy), **tr})
    at_default = [r for r in per if r["beta"] == dbeta]
    objective = statistics.mean(r["V"] for r in at_default) if at_default else 0.0
    return {
        "policy": str(policy),
        "default_beta": dbeta,
        "objective_V": round(objective, 4),
        "cost_per_attempt": cost,
        "parallel_bonus": bonus,
        "noise_pct": noise,
        "rounds": [r.round for r in rounds],
        "sweep": {
            str(b): {
                "mean_V": round(statistics.mean(r["V"] for r in per if r["beta"] == b), 4),
                "mean_attempts": statistics.mean(r["attempts"] for r in per if r["beta"] == b),
                "mean_best": round(statistics.mean(r["best"] for r in per if r["beta"] == b), 4),
            }
            for b in betas
        },
        "episodes": per,
    }
