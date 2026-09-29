#!/usr/bin/env python3
"""Fetch the inline review threads of chosen PRs (the defects reviewers caught before merge), in parallel.

  fetch_reviews.py <owner/name> <pr_numbers.txt> <out.jsonl> [--jobs 6]

One line per PR: {number, title, state, threads: [{path, line, resolved, outdated, comments: [{author, body, hunk}]}]}.
Bot accounts are dropped. The review workflow reads these.
"""
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine")
)
import spawn  # noqa: E402

repo, listfile, out = sys.argv[1:4]
jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 6
owner, name = repo.split("/")
BOT = re.compile(
    r"(\[bot\]$|bot$|^github-actions|^copilot|^coderabbit|^codecov|^dependabot)", re.I
)
# Both connections are paginated: the most-reviewed PRs are the most valuable input, and a truncated thread can hide
# the reply that says whether the defect was addressed or disputed.
Q = """query($o:String!,$n:String!,$p:Int!,$c:String){repository(owner:$o,name:$n){pullRequest(number:$p){number title
 state author{login} reviewThreads(first:50,after:$c){pageInfo{hasNextPage endCursor} nodes{id path line originalLine
  isResolved isOutdated comments(first:50){pageInfo{hasNextPage endCursor} nodes{author{login} body diffHunk}}}}}}}"""
QC = """query($id:ID!,$c:String){node(id:$id){... on PullRequestReviewThread{comments(first:50,after:$c){
 pageInfo{hasNextPage endCursor} nodes{author{login} body diffHunk}}}}}"""
nums = [int(x) for x in open(listfile).read().split() if x.strip().isdigit()]
done = set()
if os.path.exists(out):
    done = {json.loads(ln)["number"] for ln in open(out) if ln.strip()}


def gql(query, fields):
    """One GraphQL call with retries; None when it keeps failing."""
    args = ["api", "graphql", "-f", f"query={query}"]
    for k, v in fields.items():
        if v is not None:
            args += ["-F" if isinstance(v, int) else "-f", f"{k}={v}"]
    for t in range(8):
        r = spawn.run("gh", args, capture_output=True, text=True)
        if r.returncode == 0 and '"errors"' not in r.stdout[:300]:
            return json.loads(r.stdout)["data"]
        time.sleep(
            120 if "rate limit" in (r.stderr + r.stdout).lower() else 10 * (t + 1)
        )
    return None


def one(n):
    pr, nodes, cur = None, [], None
    while True:
        d = gql(Q, {"o": owner, "n": name, "p": n, "c": cur})
        if d is None:
            return None
        page = d["repository"]["pullRequest"]
        pr = pr or page
        rt = page["reviewThreads"]
        nodes += rt["nodes"]
        if not rt["pageInfo"]["hasNextPage"]:
            break
        cur = rt["pageInfo"]["endCursor"]
    for th in nodes:
        cm = th["comments"]
        while cm["pageInfo"]["hasNextPage"]:
            d = gql(QC, {"id": th["id"], "c": cm["pageInfo"]["endCursor"]})
            if d is None:
                return None
            more = d["node"]["comments"]
            cm["nodes"] += more["nodes"]
            cm["pageInfo"] = more["pageInfo"]
    threads = []
    for th in nodes:
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
failed = []
with open(out, "a") as fh, ThreadPoolExecutor(jobs) as ex:
    for i, (n, res) in enumerate(zip(todo, ex.map(one, todo))):
        if res:
            fh.write(json.dumps(res) + "\n")
            fh.flush()
        else:
            failed.append(n)
        if i % 100 == 0:
            print(f"{i}/{len(todo)}", flush=True)
print(f"done: {len(todo) - len(failed)} of {len(todo)} fetched into {out}")
if failed:
    # rerunning the same command retries exactly these: fetched PRs are skipped
    sys.exit(f"INCOMPLETE: {len(failed)} PR(s) failed after retries: {failed[:20]}")
