#!/usr/bin/env python3
"""Run the active exploration policy for the current round.

    policy_step.py --campaign C --round 1 --plan [--root <ref>]   # start a round: writes rounds/r01/manifest.json
    policy_step.py --campaign C --round 1 --next                  # next batch: logs the decision, prints the items

--next refuses to run while nodes from the previous step are neither committed nor
recorded (verify_node.py --record marks them lost/overridden). The view the policy sees is
built from git tags + the ledger, exactly as replay.py will show it later.
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dream.campaign import load_campaign, node_id  # noqa: E402
from dream.policy_api import ROOT, PlanContext, load_policy, validate  # noqa: E402
from dream.tree import load_round, online_view, round_summary  # noqa: E402


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def policy_beta(path: Path, fallback: float) -> float:
    for line in path.read_text().splitlines():
        if line.startswith("DEFAULT_BETA"):
            return float(line.split("=", 1)[1].split("#")[0])
    return fallback


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--next", action="store_true")
    ap.add_argument("--root", help="round root ref (default: campaign root)")
    args = ap.parse_args()
    c = load_campaign(args.campaign)
    rdir = c.ledger / "rounds" / f"r{args.round:02d}"
    defaults = c.cfg.get("policy_defaults", {})

    if args.plan:
        if (rdir / "manifest.json").exists():
            sys.exit(f"{rdir}/manifest.json already exists")
        version = (c.ledger / "policies" / "ACTIVE").read_text().strip()
        ppath = c.ledger / "policies" / version / "policy.py"
        beta = policy_beta(ppath, float(defaults.get("beta", 0.6)))
        pol = load_policy(ppath, {"beta": beta, "defaults": defaults})
        history = [round_summary(load_round(c, r)) for r in range(1, args.round)]
        plan = pol.plan(PlanContext(args.round, defaults, history))
        root_ref = args.root or c.ref_root()
        manifest = {
            "round": args.round,
            "policy": version,
            "beta": beta,
            "W": plan.W,
            "R": plan.R,
            "reason": plan.reason,
            "round_root": root_ref,
            "round_root_commit": c.git("rev-parse", f"{root_ref}^{{commit}}"),
            "max_steps": int(defaults.get("max_steps", 30)),
            "started_at": now(),
        }
        rdir.mkdir(parents=True, exist_ok=True)
        (rdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (rdir / "decisions.jsonl").touch()
        print(json.dumps(manifest, indent=2))
        return

    if not args.next:
        sys.exit("pass --plan or --next")
    manifest = json.loads((rdir / "manifest.json").read_text())
    rr = load_round(c, args.round)
    # every node assigned in the previous step must be committed or recorded as lost
    tagged = {n.node_id for n in rr.nodes}
    open_nodes: set[str] = set()
    for d in rr.decisions:
        if d.get("type") == "decision":
            open_nodes = {it["node"] for it in d.get("batch", [])}
        elif d.get("type") in ("lost", "override"):
            open_nodes.discard(d["node"])
    pending = sorted(open_nodes - tagged)
    if pending:
        sys.exit(f"step not finished: {pending} have no commit; run verify_node.py --record for each")
    if rr.steps_done() >= manifest["max_steps"]:
        print(json.dumps({"stop": True, "why": "max_steps reached", "items": []}))
        return

    base = c.ledger / "baseline.json"
    noise = json.loads(base.read_text())["noise_pct"] if base.exists() else 2.0
    view = online_view(rr, manifest["W"], manifest["R"], noise)
    ppath = c.ledger / "policies" / manifest["policy"] / "policy.py"
    pol = load_policy(ppath, {"beta": manifest["beta"], "defaults": defaults})
    batch = pol.select_batch(view)
    errs = validate(view, batch)
    if errs:
        sys.exit(f"policy {manifest['policy']} returned an illegal batch: {errs}")

    next_branch = max(view.branches, default=0) + 1
    # branches whose a01 was lost keep their number free; reuse it
    lost_a01 = sorted(
        int(d["node"].split("-")[1][1:]) for d in rr.decisions if d.get("type") == "lost" and d["node"].endswith("-a01")
    )
    lost_a01 = [b for b in lost_a01 if b not in view.branches]
    items = []
    for it in batch.items:
        if it.parent == ROOT:
            b = lost_a01.pop(0) if lost_a01 else next_branch
            if b == next_branch:
                next_branch += 1
            nid, wt = node_id(args.round, b, 1), c.worktree(args.round, b)
        else:
            head = next(n for n in view.nodes if n.node_id == it.parent)
            nid, wt = node_id(args.round, head.branch, head.attempt + 1), c.worktree(args.round, head.branch)
        items.append({"node": nid, "parent": it.parent, "role": it.role, "why": it.why, "worktree": str(wt)})

    decision = {
        "type": "decision",
        "round": args.round,
        "step": view.steps_done + 1,
        "time": now(),
        "policy": manifest["policy"],
        "attempts_so_far": view.attempts,
        "eligible": view.legal(),
        "batch": [{k: i[k] for k in ("node", "parent", "role", "why")} for i in items],
        "closed": {str(k): v for k, v in batch.closed.items()},
        "stop": not items,
    }
    with open(rdir / "decisions.jsonl", "a") as f:
        f.write(json.dumps(decision) + "\n")
    tools = Path(__file__).resolve().parent
    for i in items:
        i["prepare"] = f"{tools}/prepare_worker.sh --campaign {c.name} --node {i['node']} --parent {i['parent']}"
    print(
        json.dumps(
            {"stop": not items, "step": decision["step"], "closed": decision["closed"], "items": items}, indent=2
        )
    )


if __name__ == "__main__":
    main()
