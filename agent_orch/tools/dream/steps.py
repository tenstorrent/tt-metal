"""Run the active exploration policy online: plan a round, then ask it for each step's batch.

The view the policy sees is rebuilt from refs/dream/<c>/n/* and the ledger, exactly as replay.py shows
it later, which is what keeps every round replayable.
"""

from __future__ import annotations

import json
from pathlib import Path

from .campaign import Campaign, node_id, round_root_mode
from .gitops import append_jsonl, now, round_manifest
from .policy_api import ROOT, PlanContext, load_policy, validate
from .tree import load_round, online_view, round_summary


def active_version(c: Campaign) -> str:
    return (c.ledger / "policies" / "ACTIVE").read_text().strip()


def policy_beta(path: Path, fallback: float) -> float:
    for line in path.read_text().splitlines():
        if line.startswith("DEFAULT_BETA"):
            return float(line.split("=", 1)[1].split("#")[0])
    return fallback


def defaults(c: Campaign) -> dict:
    s = c.cfg["search"]
    return {
        "W": int(s["W"]),
        "R": int(s["R"]),
        "beta": float(s["beta"]),
        "max_steps": int(s["max_steps"]),
        "round_root": round_root_mode(c.cfg),  # origin | best: where the next round starts (policies may use it)
    }


def plan_round(c: Campaign, rnd: int, root_ref: str, extra: dict | None = None) -> dict:
    rdir = c.ledger / "rounds" / f"r{rnd:02d}"
    if (rdir / "manifest.json").exists():
        raise RuntimeError(f"{rdir}/manifest.json already exists")
    version = active_version(c)
    ppath = c.ledger / "policies" / version / "policy.py"
    d = defaults(c)
    beta = policy_beta(ppath, d["beta"])
    pol = load_policy(ppath, {"beta": beta, "defaults": d})
    history = [round_summary(load_round(c, r)) for r in range(1, rnd)]
    plan = pol.plan(PlanContext(rnd, d, history))
    manifest = {
        "round": rnd,
        "policy": version,
        "beta": beta,
        "W": plan.W,
        "R": plan.R,
        "reason": plan.reason,
        "round_root": root_ref,
        "round_root_commit": c.git("rev-parse", f"{root_ref}^{{commit}}"),
        "max_steps": d["max_steps"],
        "started_at": now(),
        **(extra or {}),
    }
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (rdir / "decisions.jsonl").touch()
    return manifest


def pending_items(c: Campaign, rnd: int) -> list[dict]:
    """Items of the last logged step that have neither a node ref nor a lost/override record."""
    rr = load_round(c, rnd)
    tagged = {n.node_id for n in rr.nodes}
    last, recorded = None, set()
    for d in rr.decisions:
        if d.get("type") == "decision":
            last, recorded = d, set()
        elif d.get("type") in ("lost", "override"):
            recorded.add(d["node"])
    if not last:
        return []
    return [i for i in last.get("items_full", last.get("batch", [])) if i["node"] not in tagged | recorded]


def next_batch(c: Campaign, rnd: int, max_items: int | None = None, budget_note: str = "") -> dict:
    """Ask the active policy for the next batch, log the decision. Returns {stop, step, closed, items, why}."""
    rdir = c.ledger / "rounds" / f"r{rnd:02d}"
    manifest = round_manifest(c, rnd)
    if pending_items(c, rnd):
        raise RuntimeError("step not finished: run its workers (or record them lost) first")
    rr = load_round(c, rnd)
    if rr.steps_done() >= manifest["max_steps"]:
        return {"stop": True, "why": "max_steps reached", "items": [], "closed": {}}
    base = c.ledger / "baseline.json"
    noise = json.loads(base.read_text())["noise_pct"] if base.exists() else 2.0
    view = online_view(rr, manifest["W"], manifest["R"], noise)
    ppath = c.ledger / "policies" / manifest["policy"] / "policy.py"
    pol = load_policy(ppath, {"beta": manifest["beta"], "defaults": defaults(c)})
    batch = pol.select_batch(view)
    errs = validate(view, batch)
    if errs:
        raise RuntimeError(f"policy {manifest['policy']} returned an illegal batch: {errs}")

    trimmed = 0
    if max_items is not None and len(batch.items) > max_items:
        trimmed = len(batch.items) - max_items
        batch.items = batch.items[:max_items]

    abandoned = rr.abandoned()
    next_branch = max(set(view.branches) | abandoned, default=0) + 1
    lost_a01 = sorted(
        {
            int(d["node"].split("-")[1][1:])
            for d in rr.decisions
            if d.get("type") == "lost" and d["node"].endswith("-a01")
        }
    )
    lost_a01 = [b for b in lost_a01 if b not in view.branches and b not in abandoned]
    items = []
    for it in batch.items:
        if it.parent == ROOT:
            b = lost_a01.pop(0) if lost_a01 else next_branch
            if b == next_branch:
                next_branch += 1
            nid = node_id(rnd, b, 1)
        else:
            head = next(n for n in view.nodes if n.node_id == it.parent)
            nid = node_id(rnd, head.branch, head.attempt + 1)
        items.append({"node": nid, "parent": it.parent, "role": it.role, "why": it.why})

    stop = not items
    why = "policy returned an empty batch" if not batch.items else ""
    if trimmed and stop:
        why = f"budget: {budget_note}"
    decision = {
        "type": "decision",
        "round": rnd,
        "step": view.steps_done + 1,
        "time": now(),
        "policy": manifest["policy"],
        "attempts_so_far": view.attempts,
        "eligible": view.legal(),
        "batch": items,
        "closed": {str(k): v for k, v in batch.closed.items()},
        "stop": stop,
    }
    if trimmed:
        decision["budget_trimmed"] = {"dropped_items": trimmed, "why": budget_note}
    append_jsonl(rdir / "decisions.jsonl", decision)
    return {"stop": stop, "step": decision["step"], "closed": decision["closed"], "items": items, "why": why}
