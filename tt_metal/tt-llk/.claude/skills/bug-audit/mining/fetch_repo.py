#!/usr/bin/env python3
"""Fetch ALL closed issues and ALL PRs of a GitHub repo as JSONL, in parallel weekly windows (resumable).

  fetch_repo.py <owner/name> <out_dir> [--issues-only | --prs-only] [--jobs 6]

Writes <out_dir>/issue/<window>.jsonl and <out_dir>/pr/<window>.jsonl. GitHub search returns at most 1000 results
per query, so each window is one week, and a window that still overflows is split in half automatically.
Finished windows are skipped on rerun, so a killed fetch resumes where it stopped.
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
start, end = dt.date.fromisoformat(created[:10]), dt.date.today() + dt.timedelta(days=1)
kinds = [("issue", "is:closed"), ("pr", "")]
if "--issues-only" in sys.argv:
    kinds = kinds[:1]
if "--prs-only" in sys.argv:
    kinds = kinds[1:]


def run(kind, quals, a, b):
    out = f"{outdir}/{kind}/{a}_{b}.jsonl"
    if os.path.exists(out):
        return
    r = subprocess.run(
        ["python3", f"{here}/fetch_search.py", repo, kind, quals, f"{a}..{b}", out],
        capture_output=True,
        text=True,
    )
    if r.returncode == 3 and a < b:
        mid = a + (b - a) // 2
        run(kind, quals, a, mid)
        run(kind, quals, mid + dt.timedelta(days=1), b)
    elif r.returncode != 0:
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
