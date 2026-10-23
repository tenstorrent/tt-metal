#!/usr/bin/env python3
"""Split cases.jsonl into compact digest chunks for the triage workflow, and print its args.

  make_chunks.py --cases cases.jsonl --out-dir chunks/ --repo owner/name [--size 40] [--only-with-fix]

Each chunk file holds <= --size case digests: title, labels, a trimmed body, the fix's files and size, and the later
history counts (reverts, re-fixes). The triage agent reads one chunk file; nothing else is needed to triage.
"""
import argparse
import json
import os

p = argparse.ArgumentParser()
p.add_argument("--cases", required=True)
p.add_argument("--out-dir", required=True)
p.add_argument("--repo", required=True)
p.add_argument("--size", type=int, default=40)
p.add_argument("--body", type=int, default=1500)
p.add_argument("--only-with-fix", action="store_true")
a = p.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
cases = [json.loads(x) for x in open(a.cases) if x.strip()]
if a.only_with_fix:
    cases = [c for c in cases if c["fix_oids"] or c.get("xref_fix")]


def digest(c):
    files = sorted({f for fx in c["fix"] for f in fx["files"]})
    later = c.get("later") or {}
    return {
        "id": c["id"],
        "title": c["title"],
        "labels": c["labels"],
        "signal": c["bug_signal"],
        "created": c["created"][:10],
        "body": " ".join((c.get("body") or "").split())[: a.body],
        "fix_subjects": [fx["subject"] for fx in c["fix"]][:5]
        + [x["subject"] for x in c.get("xref_fix", [])][:3],
        "fix_files": files[:25],
        "fix_nfiles": len(files),
        "fix_size": sum(fx["adds"] + fx["dels"] for fx in c["fix"]),
        "reverted": bool(later.get("reverts")),
        "later_citing": len(later.get("citing_later", [])),
        "later_fixlike_same_files": later.get("fixlike_same_files_count", 0),
        "review_threads": c.get("review_threads", 0),
    }


chunks = []
for i in range(0, len(cases), a.size):
    path = os.path.join(os.path.abspath(a.out_dir), f"chunk-{i // a.size:04d}.json")
    with open(path, "w") as fh:
        json.dump(
            {"repo": a.repo, "cases": [digest(c) for c in cases[i : i + a.size]]}, fh
        )
    chunks.append(path)
print(json.dumps({"repo": a.repo, "chunks": chunks}))
