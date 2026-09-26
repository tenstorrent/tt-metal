#!/usr/bin/env python3
"""Fetch the inline review threads of chosen PRs (the defects reviewers caught before merge), in parallel.

  fetch_reviews.py <owner/name> <pr_numbers.txt> <out.jsonl> [--jobs 6]

One line per PR: {number, title, state, threads: [{path, line, resolved, outdated, comments: [{author, body, hunk}]}]}.
Bot accounts are dropped. The review workflow reads these.
"""
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

repo, listfile, out = sys.argv[1:4]
jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 6
owner, name = repo.split("/")
BOT = re.compile(
    r"(\[bot\]$|bot$|^github-actions|^copilot|^coderabbit|^codecov|^dependabot)", re.I
)
Q = """query($o:String!,$n:String!,$p:Int!){repository(owner:$o,name:$n){pullRequest(number:$p){number title state
 author{login} reviewThreads(first:60){nodes{path line originalLine isResolved isOutdated
  comments(first:12){nodes{author{login} body diffHunk}}}}}}}"""
nums = [int(x) for x in open(listfile).read().split() if x.strip().isdigit()]
done = set()
if os.path.exists(out):
    done = {json.loads(ln)["number"] for ln in open(out) if ln.strip()}


def one(n):
    for t in range(8):
        r = subprocess.run(
            [
                "gh",
                "api",
                "graphql",
                "-f",
                f"query={Q}",
                "-F",
                f"o={owner}",
                "-F",
                f"n={name}",
                "-F",
                f"p={n}",
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode == 0 and '"errors"' not in r.stdout[:300]:
            break
        time.sleep(
            120 if "rate limit" in (r.stderr + r.stdout).lower() else 10 * (t + 1)
        )
    else:
        return None
    pr = json.loads(r.stdout)["data"]["repository"]["pullRequest"]
    threads = []
    for th in pr["reviewThreads"]["nodes"]:
        cs = [
            {
                "author": (c["author"] or {}).get("login", "?"),
                "body": c["body"][:1500],
                "hunk": (c.get("diffHunk") or "")[-800:],
            }
            for c in th["comments"]["nodes"]
            if not BOT.search((c["author"] or {}).get("login", ""))
        ]
        if cs:
            threads.append(
                {
                    "path": th["path"],
                    "line": th["line"] or th["originalLine"],
                    "resolved": th["isResolved"],
                    "outdated": th["isOutdated"],
                    "comments": cs,
                }
            )
    return {
        "number": pr["number"],
        "title": pr["title"],
        "state": pr["state"],
        "author": (pr["author"] or {}).get("login"),
        "threads": threads,
    }


todo = [n for n in nums if n not in done]
with open(out, "a") as fh, ThreadPoolExecutor(jobs) as ex:
    for i, res in enumerate(ex.map(one, todo)):
        if res:
            fh.write(json.dumps(res) + "\n")
            fh.flush()
        if i % 100 == 0:
            print(f"{i}/{len(todo)}", flush=True)
print(f"done: {len(todo)} fetched into {out}")
