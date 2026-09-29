#!/usr/bin/env python3
"""Fetch ALL closed issues and ALL PRs of a GitHub repo as JSONL, in parallel weekly windows (resumable).

  fetch_repo.py <owner/name> <out_dir> [--issues-only | --prs-only] [--jobs 6] [--since YYYY-MM-DD]

Writes <out_dir>/issue/<window>.jsonl and <out_dir>/pr/<window>.jsonl. GitHub search returns at most 1000 results
per query, so each window is one week, and a window that still overflows is split in half automatically.
Finished windows are skipped on rerun, so a killed fetch resumes where it stopped.

`--since` is the incremental refresh: it fetches only what CLOSED on or after that date (closed issues, closed and
merged PRs), windowed on the close date, into closed_<window>.jsonl files next to the full fetch. Windowing on the
creation date would miss every long-lived bug closed since the last mining. Take the date from the store's marker
(mining/marker.py). A window that reaches today is always refetched, since more will close in it.
"""
import datetime as dt
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

here = os.path.dirname(os.path.abspath(__file__))
repo, outdir = sys.argv[1:3]
jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 6
created = json.loads(
    subprocess.run(
        ["gh", "api", f"repos/{repo}"], capture_output=True, text=True, check=True
    ).stdout
)["created_at"]
since = sys.argv[sys.argv.index("--since") + 1] if "--since" in sys.argv else None
start = dt.date.fromisoformat(since or created[:10])
end = dt.date.today() + dt.timedelta(days=1)
field, prefix = ("closed", "closed_") if since else ("created", "")
# a refresh wants what closed: open PRs are not fixes yet
kinds = [("issue", "is:closed"), ("pr", "is:closed" if since else "")]
if "--issues-only" in sys.argv:
    kinds = kinds[:1]
if "--prs-only" in sys.argv:
    kinds = kinds[1:]


FAILED = (
    []
)  # windows that could not be fetched (or split): the dump is incomplete until they are rerun


def run(kind, quals, a, b):
    out = f"{outdir}/{kind}/{prefix}{a}_{b}.jsonl"
    # complete only if it was fetched AFTER the window ended: one saved while the window was still open (a rerun on a
    # later day, a resumed full fetch) is missing whatever closed or opened after that, and is fetched again
    if os.path.exists(out) and dt.date.fromtimestamp(os.path.getmtime(out)) > b:
        return
    r = subprocess.run(
        [
            "python3",
            f"{here}/fetch_search.py",
            repo,
            kind,
            quals,
            f"{a}..{b}",
            out,
            "--field",
            field,
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode == 3 and a < b:
        mid = a + (b - a) // 2
        run(kind, quals, a, mid)
        run(kind, quals, mid + dt.timedelta(days=1), b)
    elif r.returncode != 0:
        FAILED.append((kind, str(a), str(b)))
        print("FAIL", kind, a, b, r.stderr[-300:], flush=True)
    else:
        print(kind, r.stdout.strip(), flush=True)


for kind, quals in kinds:
    os.makedirs(f"{outdir}/{kind}", exist_ok=True)
    wins, d = [], start
    while d <= end:
        e = min(d + dt.timedelta(days=6), end)
        wins.append((d, e))
        d = e + dt.timedelta(days=1)
    with ThreadPoolExecutor(jobs) as ex:
        list(ex.map(lambda w: run(kind, quals, *w), wins))
    print("done", kind)
if FAILED:
    # a missing window silently drops its issues/PRs from triage, class weights and the holdout
    sys.exit(
        f"INCOMPLETE: {len(FAILED)} window(s) failed; rerun the same command to retry them: {FAILED[:10]}"
    )
