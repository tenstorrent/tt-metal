#!/usr/bin/env python3
"""Orchestrator's check of a returned worker node.

    verify_node.py --campaign C --node N [--record]

Checks the tag exists, its git parent is the expected parent (round root or parent tag), node.json
and score.json agree, and the commit only touches allowed paths + its own node dir. With --record,
a failing node gets an override line in the round's decisions.jsonl (the commit is never rewritten).
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dream.campaign import load_campaign, parse_node_id  # noqa: E402
from dream.tree import read_node_file  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--node", required=True)
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args()
    c = load_campaign(args.campaign)
    rnd, _, att = parse_node_id(args.node)
    issues, fail_class = [], "infra"

    try:
        sha = c.git("rev-parse", f"{c.ref_node(args.node)}^{{commit}}")
    except RuntimeError:
        issues.append("tag missing (worker did not commit)")
        sha = None
    if sha:
        manifest = json.loads((c.ledger / "rounds" / f"r{rnd:02d}" / "manifest.json").read_text())
        meta = json.loads(read_node_file(c, args.node, "node.json") or "{}")
        score = json.loads(read_node_file(c, args.node, "eval/score.json") or "{}")
        exp_parent = (
            manifest["round_root_commit"]
            if att == 1
            else c.git("rev-parse", f"{c.ref_node(meta.get('parent', ''))}^{{commit}}") if meta.get("parent") else None
        )
        if c.git("rev-parse", f"{sha}^") != exp_parent:
            issues.append(f"git parent is not the expected parent {meta.get('parent')}")
        if not meta or not score:
            issues.append("node.json or eval/score.json missing")
        for k in ("valid", "fail_class", "score"):
            if meta.get(k) != score.get(k):
                issues.append(f"node.json {k}={meta.get(k)} != score.json {score.get(k)}")
        files = c.git("diff", "--name-only", f"{sha}^", sha).split()
        bad = [p for p in files if not c.allowed(p, args.node)]
        if bad:
            issues.append(f"touches files outside allowed_paths: {bad}")
            fail_class = "forbidden_edit"
    else:
        fail_class = "lost"

    ok = not issues
    if args.record and not ok:
        path = c.ledger / "rounds" / f"r{rnd:02d}" / "decisions.jsonl"
        line = {
            "type": "lost" if fail_class == "lost" else "override",
            "node": args.node,
            "fail_class": fail_class,
            "why": "; ".join(issues),
            "time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }
        with open(path, "a") as f:
            f.write(json.dumps(line) + "\n")
    print(json.dumps({"node": args.node, "ok": ok, "issues": issues}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
