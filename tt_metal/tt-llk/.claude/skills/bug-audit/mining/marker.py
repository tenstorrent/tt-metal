#!/usr/bin/env python3
"""The mining watermark: how far a repo's history has been mined, so a refresh reads only what is new.

  marker.py write <marker.json> --repo owner/name --dumps 'glob[,glob...]' [--tree-commit SHA]
                  [--deep deep.jsonl[,...]] [--holdout holdout.jsonl[,...]] [--public]
  marker.py delta <marker.json>     # count what closed since the watermark (count queries only, no fetch)
  marker.py show  <marker.json>

The watermark is the latest close time (closedAt / mergedAt) in the dumps -- a CLOSE time, not a creation time,
because a refresh must also pick up the long-lived bugs that closed since. The marker also records the tree the deep
reads judged ("current tree"), the counts, and the ids already deep-read or held out, so a refresh can tell new cases
from old ones without reading the old ones again.

Keep one marker with the mined store (<mine>/<repo>.mined.json), rewritten after every refresh. A published pack gets
a marker of its own next to it (packs/<repo>.mined.json), written with --public: dates, a commit and counts only, no
issue or PR ids. Leave --tree-commit out unless you know which tree the deep reads judged; a wrong commit is worse
than none.
"""
import datetime as dt
import glob
import json
import subprocess
import sys

argv = sys.argv[1:]
if len(argv) < 2:
    sys.exit(__doc__)
cmd, path = argv[0], argv[1]


def opt(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def files(spec):
    return sorted({f for g in (spec or "").split(",") if g for f in glob.glob(g)})


def ids(spec, field="id"):
    return sorted(
        {json.loads(ln)[field] for f in files(spec) for ln in open(f) if ln.strip()}
    )


if cmd == "write":
    repo = opt("--repo")
    dumps = files(opt("--dumps"))
    if not repo or not dumps:
        sys.exit("write needs --repo and --dumps matching at least one file")
    latest, n_issue, n_pr = "", 0, 0
    for f in dumps:
        for ln in open(f):
            if not ln.strip():
                continue
            x = json.loads(ln)
            is_pr = "mergedAt" in x or "mergeCommit" in x
            n_pr += is_pr
            n_issue += not is_pr
            latest = max(latest, x.get("mergedAt") or "", x.get("closedAt") or "")
    if not latest:
        sys.exit("no closed item in the dumps: nothing to mark")
    marker = {
        "repo": repo,
        "watermark": latest,
        "tree_commit": opt("--tree-commit"),
        "written": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "counts": {"issues": n_issue, "prs": n_pr, "dump_files": len(dumps)},
        "deep_read_ids": ids(opt("--deep")),
        "holdout_ids": ids(opt("--holdout")),
    }
    if "--public" in argv:
        marker["counts"]["deep_read"] = len(marker.pop("deep_read_ids"))
        marker["counts"]["held_out"] = len(marker.pop("holdout_ids"))
    with open(path, "w") as fh:
        json.dump(marker, fh, indent=1)
    c = marker["counts"]
    n_deep = c.get("deep_read", len(marker.get("deep_read_ids", [])))
    n_hold = c.get("held_out", len(marker.get("holdout_ids", [])))
    print(
        f"{repo}: watermark {latest} ({n_issue} issues, {n_pr} PRs, {n_deep} deep-read, {n_hold} held out) -> {path}"
    )
elif cmd in ("delta", "show"):
    m = json.load(open(path))
    since = m["watermark"][:10]
    c = m.get("counts", {})
    n_deep = c.get("deep_read", len(m.get("deep_read_ids", [])))
    n_hold = c.get("held_out", len(m.get("holdout_ids", [])))
    print(
        f"{m['repo']}: mined through {m['watermark']} (tree {m.get('tree_commit') or 'not recorded'}; "
        f"{n_deep} deep-read, {n_hold} held out; written {m['written']})"
    )
    if cmd == "delta":
        # the watermark day is re-counted on purpose: the next fetch starts there, and cases dedupe by number
        q = "query($q:String!){search(type:ISSUE,first:1,query:$q){issueCount}}"
        for label, quals in (
            ("issues closed", f"is:issue is:closed closed:>={since}"),
            ("PRs merged", f"is:pr is:merged merged:>={since}"),
            ("PRs closed unmerged", f"is:pr is:closed is:unmerged closed:>={since}"),
        ):
            r = subprocess.run(
                [
                    "gh",
                    "api",
                    "graphql",
                    "-f",
                    f"query={q}",
                    "-f",
                    f"q=repo:{m['repo']} {quals}",
                ],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                sys.exit(f"count query failed: {(r.stderr or r.stdout)[:200]}")
            print(
                f"  {label} since {since}: {json.loads(r.stdout)['data']['search']['issueCount']}"
            )
        print(
            f"  refresh: fetch_repo.py {m['repo']} <mine>/raw --since {since}, then build/triage/deep-read the new cases"
        )
else:
    sys.exit(__doc__)
