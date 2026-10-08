#!/usr/bin/env python3
"""Replay exploration policies on recorded rounds ("dreaming").

    replay.py --campaign rmsnorm-prefill --policy <dir>/policy.py [--policy ...]
              [--rounds 1,2] [--beta-sweep 0.2,0.4,...] [--out replay.json] [--traces traces.jsonl]
    replay.py --fixture tests/fixture_example.fixture --policy ... --cost 0.005 --bonus 0.01

For each policy, beta and recorded round, the policy starts seeing only the
round root. Each batch reveals the recorded child of every selected start
point (root -> earliest unopened recorded branch, head -> that branch's next
recorded attempt). Nothing is generated or run. The score of one replay is

    V = best valid score revealed - cost * attempts revealed + bonus * attempts / steps

and a policy's objective is V averaged over rounds at its default beta.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dream.campaign import load_campaign  # noqa: E402
from dream.policy_api import (  # noqa: E402
    ROOT,
    PlanContext,
    RoundView,
    load_policy,
    replay_value,
    validate,
)
from dream.tree import (  # noqa: E402
    RecordedRound,
    load_fixture,
    load_round,
    recorded_rounds,
    round_summary,
)


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
    view = RoundView(rr.round, plan.W, plan.R, noise_pct, {}, {}, 0)
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--campaign")
    src.add_argument("--fixture", type=Path)
    ap.add_argument("--policy", type=Path, action="append", required=True)
    ap.add_argument("--rounds", help="comma-separated round numbers (default: all recorded)")
    ap.add_argument("--beta-sweep", help="comma-separated betas (default: campaign dreaming.beta_sweep)")
    ap.add_argument("--cost", type=float)
    ap.add_argument("--bonus", type=float)
    ap.add_argument("--out", type=Path, help="write replay.json (single policy) or a {policy: result} map")
    ap.add_argument("--traces", type=Path)
    args = ap.parse_args()

    if args.campaign:
        c = load_campaign(args.campaign)
        dcfg, defaults = c.cfg.get("dreaming", {}), c.cfg.get("policy_defaults", {})
        wanted = [int(x) for x in args.rounds.split(",")] if args.rounds else recorded_rounds(c)
        rounds = [load_round(c, r) for r in wanted]
        base = c.ledger / "baseline.json"
        noise = json.loads(base.read_text())["noise_pct"] if base.exists() else 2.0
    else:
        dcfg, defaults = {}, {"W": 4, "R": 4, "beta": 0.6}
        rounds, noise = load_fixture(args.fixture)
        if args.rounds:
            keep = {int(x) for x in args.rounds.split(",")}
            rounds = [r for r in rounds if r.round in keep]
        for r in rounds:  # fixtures carry their own grid size
            if r.manifest.get("W"):
                defaults = {**defaults, "W": r.manifest["W"], "R": r.manifest["R"]}
    cost = args.cost if args.cost is not None else float(dcfg.get("cost_per_attempt", 0.005))
    bonus = args.bonus if args.bonus is not None else float(dcfg.get("parallel_bonus", 0.01))
    sweep = (
        [float(x) for x in args.beta_sweep.split(",")]
        if args.beta_sweep
        else [float(x) for x in dcfg.get("beta_sweep", [0.6])]
    )
    max_steps = int(dcfg.get("max_replay_steps", 100))

    all_results, traces = {}, []
    for pp in args.policy:
        dbeta = policy_default_beta(pp, float(defaults.get("beta", 0.6)))
        betas = sorted(set(sweep) | {dbeta})
        per = []
        for beta in betas:
            for i, rr in enumerate(rounds):
                history = [round_summary(r) for r in rounds[:i]]
                res, tr = simulate(pp, beta, rr, history, defaults, noise, cost, bonus, max_steps)
                per.append(res)
                traces.append({"policy": str(pp), **tr})
        at_default = [r for r in per if r["beta"] == dbeta]
        objective = statistics.mean(r["V"] for r in at_default) if at_default else 0.0
        sweep_summary = {
            str(b): {
                "mean_V": round(statistics.mean(r["V"] for r in per if r["beta"] == b), 4),
                "mean_attempts": statistics.mean(r["attempts"] for r in per if r["beta"] == b),
                "mean_best": round(statistics.mean(r["best"] for r in per if r["beta"] == b), 4),
            }
            for b in betas
        }
        all_results[str(pp)] = {
            "policy": str(pp),
            "default_beta": dbeta,
            "objective_V": round(objective, 4),
            "cost_per_attempt": cost,
            "parallel_bonus": bonus,
            "noise_pct": noise,
            "rounds": [r.round for r in rounds],
            "sweep": sweep_summary,
            "episodes": per,
        }
        errs = [r for r in per if "error" in r]
        print(
            f"{pp}: objective V = {objective:.4f} at beta={dbeta}"
            + (f"  ({len(errs)} ILLEGAL episodes)" if errs else "")
        )
        for r in at_default:
            print(
                f"    r{r['round']:02d}: V={r['V']:.4f} best={r['best']:.3f} attempts={r['attempts']} steps={r['steps']}"
            )
        for e in errs[:3]:
            print(f"    ! r{e['round']:02d} beta={e['beta']}: {e['error']}")

    if args.out:
        payload = next(iter(all_results.values())) if len(all_results) == 1 else all_results
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
    if args.traces:
        args.traces.write_text("".join(json.dumps(t) + "\n" for t in traces))


if __name__ == "__main__":
    main()
