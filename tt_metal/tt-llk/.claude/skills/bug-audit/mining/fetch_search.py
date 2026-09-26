#!/usr/bin/env python3
"""Fetch issues/PRs via GraphQL search over one created: window (search caps at 1000 results per query).
usage: fetch_search.py <repo> <pr|issue> "<qualifiers>" <YYYY-MM-DD..YYYY-MM-DD> <out.jsonl>
Exit 3 = window holds >1000 results; split it and rerun."""
import json
import subprocess
import sys
import time

repo, kind, quals, win, out = sys.argv[1:6]
PR = """... on PullRequest{number title createdAt mergedAt closedAt state author{login} labels(first:15){nodes{name}}
  body additions deletions changedFiles mergeCommit{oid} closingIssuesReferences(first:10){nodes{number}}
  reviewThreads{totalCount} reviews{totalCount} comments{totalCount}}"""
IS = """... on Issue{number title createdAt closedAt stateReason state author{login} labels(first:20){nodes{name}}
  body comments{totalCount}
  timelineItems(last:10,itemTypes:[CLOSED_EVENT,CROSS_REFERENCED_EVENT,REFERENCED_EVENT]){nodes{__typename
   ... on ClosedEvent{closer{__typename ... on PullRequest{number merged title} ... on Commit{oid messageHeadline}}}
   ... on CrossReferencedEvent{source{__typename ... on PullRequest{number merged title}}}
   ... on ReferencedEvent{commit{oid messageHeadline}}}}}"""
Q = (
    "query($q:String!,$c:String){search(type:ISSUE,first:50,after:$c,query:$q)"
    "{issueCount pageInfo{hasNextPage endCursor} nodes{%s}}}"
) % (PR if kind == "pr" else IS)
q = f"repo:{repo} is:{kind} {quals} created:{win}"
cur, n, total = None, 0, 0
with open(out + ".part", "w") as f:
    while True:
        args = ["gh", "api", "graphql", "-f", f"query={Q}", "-f", f"q={q}"]
        if cur:
            args += ["-f", f"c={cur}"]
        t = 0
        while True:
            r = subprocess.run(args, capture_output=True, text=True)
            if r.returncode == 0 and '"errors"' not in r.stdout[:300]:
                break
            msg = r.stderr or r.stdout
            if "rate limit" in msg.lower() or "RATE_LIMITED" in msg:
                print("rate-limited", win, file=sys.stderr, flush=True)
                time.sleep(120)
                continue
            t += 1
            print("retry", win, msg[:200], file=sys.stderr, flush=True)
            if t > 8:
                sys.exit(1)
            time.sleep(15 * t)
        d = json.loads(r.stdout)["data"]["search"]
        total = d["issueCount"]
        if total > 1000:
            print(f"OVERFLOW {win} {total}", file=sys.stderr)
            sys.exit(3)
        for x in d["nodes"]:
            f.write(json.dumps(x) + "\n")
            n += 1
        cur = d["pageInfo"]["endCursor"]
        if not d["pageInfo"]["hasNextPage"]:
            break
import os

os.replace(out + ".part", out)
print(win, n, total)
