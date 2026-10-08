#!/usr/bin/env python3
"""Export a whole campaign (every attempt, the ledger, transcripts, a git bundle) into one folder.

    export_campaign.py --campaign rmsnorm-prefill [--out agent_orch/campaigns/<c>/export]

Run on the machine that holds the dream/<c>/* refs and $DREAM_HOME/<c>. The folder is plain
files (browsable, loadable with pandas); the bundle restores the exact git structure:

    git fetch <out>/dream_<c>.bundle 'refs/*:refs/*'
"""

import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dream.campaign import load_campaign, parse_node_id  # noqa: E402
from dream.tree import load_round, read_node_file, recorded_rounds  # noqa: E402

NODE_FILES = [
    "node.json",
    "proposal.md",
    "context.md",
    "reflection.md",
    "eval/score.json",
    "eval/summary.md",
    "eval/ops.csv",
    "eval/error.txt",
]


def run(cmd, cwd, binary=False):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, check=True)
    return r.stdout if binary else r.stdout.decode()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    c = load_campaign(args.campaign)
    repo = c.repo
    out = args.out or repo / "agent_orch" / "campaigns" / c.name / "export"
    if out.exists():
        shutil.rmtree(out)
    (out / "attempts").mkdir(parents=True)
    root = c.ref_root()
    shapes = [s["id"] for s in c.cfg["shapes"]]

    rows, lost = [], []
    for rnd in recorded_rounds(c):
        rr = load_round(c, rnd)
        overrides = {d["node"]: d for d in rr.decisions if d.get("type") == "override"}
        lost += [
            {"node_id": d["node"], "round": rnd, "why": d.get("why", "")}
            for d in rr.decisions
            if d.get("type") == "lost"
        ]
        man = rr.manifest
        for n in rr.nodes:
            nid, tag = n.node_id, c.ref_node(n.node_id)
            d = out / "attempts" / nid
            for rel in NODE_FILES:
                txt = read_node_file(c, nid, rel)
                if txt is not None:
                    (d / rel).parent.mkdir(parents=True, exist_ok=True)
                    (d / rel).write_text(txt + ("\n" if not txt.endswith("\n") else ""))
            (d / "patch.diff").write_text(run(["git", "diff", f"{tag}~1", tag, "--", ".", ":!agent_orch"], repo))
            (d / "cumulative.diff").write_text(run(["git", "diff", root, tag, "--", ".", ":!agent_orch"], repo))
            meta = json.loads(read_node_file(c, nid, "node.json") or "{}")
            score = json.loads(read_node_file(c, nid, "eval/score.json") or "{}")
            ov = overrides.get(nid)
            row = {
                "node_id": nid,
                "round": rnd,
                "branch": n.branch,
                "attempt": n.attempt,
                "parent": n.parent,
                "round_root": man.get("round_root", root),
                "policy": man.get("policy"),
                "commit": run(["git", "rev-parse", f"{tag}^{{commit}}"], repo).strip(),
                "mechanism": meta.get("mechanism", ""),
                "tags": ";".join(meta.get("tags", [])),
                "eval_valid": score.get("valid"),
                "eval_score": score.get("score"),
                "valid": n.valid,
                "fail_class": n.fail_class,
                "score": n.score,
                "override": ov.get("why") if ov else "",
                "closed_after": rr.closed().get(n.branch, ""),
            }
            for sid in shapes:
                v = score.get("shapes", {}).get(sid, {})
                for k in ("us_chip_mean", "us_chip_max", "speedup", "pcc", "max_abs"):
                    row[f"{sid}.{k}"] = v.get(k)
            rows.append(row)

    (out / "index.json").write_text(json.dumps({"campaign": c.name, "attempts": rows, "lost": lost}, indent=2) + "\n")
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ledger: exact copy of the ledger branch (policies, rounds, decisions, summaries, FINAL.md)
    tar = run(["git", "archive", c.ref_ledger()], repo, binary=True)
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(out / "ledger", filter="data")
    for extra in ("FINAL.md",):
        if (out / "ledger" / extra).exists():
            shutil.copy(out / "ledger" / extra, out / extra)
    for f in ("history.md", "tree.html"):
        if (c.home / f).exists():
            shutil.copy(c.home / f, out / f)

    # transcripts of every worker and dreaming session
    logs = sorted((c.home / "logs").glob("*.jsonl"))
    with tarfile.open(out / "transcripts.tar.gz", "w:gz") as t:
        for p in logs:
            t.add(p, arcname=f"transcripts/{p.name}")

    # git bundle: every dream/<c>/* ref, prerequisite = the campaign root's parent (on origin)
    refs = run(
        [
            "git",
            "for-each-ref",
            "--format=%(refname)",
            f"refs/tags/dream/{c.name}",
            f"refs/heads/dream/{c.name}",
        ],
        repo,
    ).split()
    bundle = out / f"dream_{c.name}.bundle"
    run(["git", "bundle", "create", str(bundle), *refs, f"^{root}^"], repo)

    best = max((r for r in rows if r["valid"]), key=lambda r: r["score"])
    (out / "README.md").write_text(
        f"""# {c.name}: full campaign export

Every attempt of the Dream-RSI campaign on `{c.cfg.get('op')}`, exported from the git refs
`dream/{c.name}/*` and `$DREAM_HOME/{c.name}` on {c.cfg.get('machine')}.

Best valid attempt: `{best['node_id']}`, score {best['score']:.4f}. See `FINAL.md`.

| Path | Contents |
|---|---|
| `FINAL.md` | final report: result, lineage of the best node, dead ends, open leads, policy evolution |
| `index.json`, `index.csv` | one row per committed attempt ({len(rows)}): round, branch, parent, policy, mechanism, validity (after overrides), score, per-shape µs / speedup / PCC / max_abs; `lost` lists attempts that never committed |
| `attempts/<node_id>/` | proposal, context, reflection, node.json, eval/ (score.json, summary.md, ops.csv), `patch.diff` (vs parent), `cumulative.diff` (vs the campaign root) |
| `ledger/` | the ledger branch: policies v0..vN with every dreaming candidate and its replay results, per-round manifest / decisions / summary, aborted rounds, baseline |
| `tree.html`, `history.md` | the visual tree and the worker-facing history, as of the end of the campaign |
| `transcripts.tar.gz` | full stream-json transcripts of every worker and policy-development session |
| `dream_{c.name}.bundle` | git bundle with every attempt commit, node tag, branch and the ledger |

`valid`/`score` in the index apply ledger overrides (e.g. attempts invalidated by a later campaign rule);
`eval_valid`/`eval_score` are what the eval tool recorded at the time.

## Restore the git structure

```bash
git fetch {out.relative_to(repo)}/dream_{c.name}.bundle 'refs/*:refs/*'
git log --oneline dream/{c.name}/root..dream/{c.name}/n/{best['node_id']}
git checkout dream/{c.name}/n/{best['node_id']}   # build + run the campaign test to reproduce
```

Raw profiler output (tracy, device logs; ~240 MB per attempt) is not included; it stays under
`$DREAM_HOME/{c.name}/reports/` on {c.cfg.get('machine')}.
"""
    )
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"exported {len(rows)} attempts ({len(lost)} lost) to {out} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
