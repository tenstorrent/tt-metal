#!/usr/bin/env python3
"""Filter fetched review threads down to the ones that might name a defect, and chunk them for review-wave.js.

  make_review_chunks.py <reviews.jsonl> <out_dir> [--budget 40000] [--path-glob 'tt_metal/*' ...]

The filter is cheap and deliberately loose: it keeps a thread when a comment uses defect vocabulary, and drops pure
nits (typo, rename, naming, formatting, nit:) and threads on docs or CI files. The agents make the real judgement. Chunks are
packed to about --budget characters so every agent gets a similar load. Prints the review-wave args skeleton.
"""
import fnmatch
import json
import os
import re
import sys

src, out = sys.argv[1:3]
budget = (
    int(sys.argv[sys.argv.index("--budget") + 1]) if "--budget" in sys.argv else 40000
)
globs = [sys.argv[i + 1] for i, x in enumerate(sys.argv) if x == "--path-glob"]
DEFECT = re.compile(
    r"\b(bug|wrong|incorrect|race|hang|deadlock|overflow|underflow|off[- ]by|uninit|leak|crash|segfault|"
    r"never|always|missing|forgot|should(n't| not)? be|won't|doesn't|does not|will not|break|broken|"
    r"mismatch|corrupt|stale|dangling|out of bounds|oob|undefined|ub\b|trunc|signed|unsigned|"
    r"barrier|flush|semaphore|sync|wait|order|alias|double|twice|leftover|edge case|zero|nan|inf|"
    r"precision|round|cache|hash|invalid|assert|guard|check)\w*",
    re.I,
)
NIT = re.compile(
    r"^\s*(nit|typo|naming|rename|format|style|nitpick)\b|^\s*(lgtm|thanks|done|fixed|ok)\b",
    re.I,
)
SKIP_PATH = re.compile(r"(\.md$|\.rst$|\.txt$|^\.github/|/docs?/|\.ya?ml$|CODEOWNERS)")
os.makedirs(out, exist_ok=True)
chunks, cur, size, kept, total = [], [], 0, 0, 0
for ln in open(src):
    pr = json.loads(ln)
    kept_threads = []
    for i, th in enumerate(pr["threads"]):
        total += 1
        if SKIP_PATH.search(th["path"] or "") or (
            globs and not any(fnmatch.fnmatch(th["path"] or "", g) for g in globs)
        ):
            continue
        first = th["comments"][0]["body"] if th["comments"] else ""
        text = " ".join(c["body"] for c in th["comments"])
        if NIT.search(first) or not DEFECT.search(text):
            continue
        kept_threads.append({"idx": i, **th})
    if not kept_threads:
        continue
    kept += len(kept_threads)
    rec = {
        "number": pr["number"],
        "title": pr["title"],
        "state": pr["state"],
        "threads": kept_threads,
    }
    n = len(json.dumps(rec))
    if cur and size + n > budget:
        chunks.append(cur)
        cur, size = [], 0
    cur.append(rec)
    size += n
if cur:
    chunks.append(cur)
paths = []
for i, ch in enumerate(chunks):
    path = os.path.join(os.path.abspath(out), f"chunk-{i:04d}.json")
    json.dump({"prs": ch}, open(path, "w"))
    paths.append(path)
print(json.dumps({"chunk_dir": os.path.abspath(out), "n_chunks": len(paths)}))
print(f"kept {kept} of {total} threads -> {len(paths)} chunks", file=sys.stderr)
