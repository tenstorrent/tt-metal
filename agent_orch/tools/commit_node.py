#!/usr/bin/env python3
"""Worker's last step: validate the node directory, sync node.json with the eval, commit and tag.

    commit_node.py --campaign C --node N          (run from inside your worktree)

Refuses to commit if required files are missing, if files outside allowed_paths changed,
or if the code changed after it was evaluated. Prints the JSON report to send back.
"""

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dream.campaign import load_campaign, repo_root  # noqa: E402

REQUIRED = ["node.json", "proposal.md", "context.md", "reflection.md", "eval/score.json"]


def git(*a, cwd):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--node", required=True)
    args = ap.parse_args()
    wt = repo_root()
    c = load_campaign(args.campaign, wt)
    rel = c.attempts_rel(args.node)
    node_dir = wt / rel

    missing = [f for f in REQUIRED if not (node_dir / f).exists()]
    if missing:
        sys.exit(f"missing in {rel}: {missing}")
    node = json.loads((node_dir / "node.json").read_text())
    score = json.loads((node_dir / "eval/score.json").read_text())
    if not node.get("mechanism"):
        sys.exit("node.json: fill in 'mechanism' (one line)")

    changed = sorted(
        set(git("diff", "--name-only", "HEAD", cwd=wt).split())
        | set(git("ls-files", "--others", "--exclude-standard", cwd=wt).split())
    )
    bad = [p for p in changed if not c.allowed(p, args.node)]
    if bad:
        sys.exit(f"files outside allowed_paths changed; revert them (git checkout -- / rm): {bad}")
    code = [p for p in changed if not p.startswith(rel + "/")]

    snap = score.get("commit_under_test")
    if snap:
        diff = subprocess.run(
            ["git", "diff", "--name-only", snap, "--", ".", f":!{c.attempts_rel()}"],
            cwd=wt,
            capture_output=True,
            text=True,
        ).stdout.split()
        if diff:
            sys.exit(f"code changed after evaluation ({diff}); re-run eval_attempt.sh before committing")

    node.update(
        valid=score.get("valid", False),
        fail_class=score.get("fail_class"),
        score=score.get("score", 0.0),
        files_changed=code,
        build_seconds=score.get("build_seconds"),
        eval_seconds=score.get("eval_seconds"),
        finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    )
    (node_dir / "node.json").write_text(json.dumps(node, indent=2) + "\n")

    outcome = f"score {node['score']:.4f}" if node["valid"] else node["fail_class"]
    msg = f"[dream:{args.campaign}] {args.node}: {node['mechanism']} ({outcome})".replace("\n", " ")
    git("add", "-A", "--", *code, rel, cwd=wt)
    git("commit", "-q", "--no-verify", "-m", msg, cwd=wt)
    sha = git("rev-parse", "HEAD", cwd=wt)
    git("tag", c.ref_node(args.node), sha, cwd=wt)

    nxt = ""
    refl = (node_dir / "reflection.md").read_text().splitlines()
    for i, line in enumerate(refl):
        if line.lower().startswith("## what a child"):
            nxt = next((l.strip("- ").strip() for l in refl[i + 1 :] if l.strip()), "")
            break
    print(
        json.dumps(
            {
                "node_id": args.node,
                "commit": sha,
                "valid": node["valid"],
                "fail_class": node["fail_class"],
                "score": node["score"],
                "mechanism": node["mechanism"],
                "next": nxt,
            }
        )
    )


if __name__ == "__main__":
    main()
