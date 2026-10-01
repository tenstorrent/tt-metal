#!/usr/bin/env python3
"""Fold a finished mining workflow's output into its durable store (upsert by case id).

  persist_mining.py <raw_output.json> <store.jsonl> [--key id]

Works for the triage, deep-read and review workflows: each returns {results: [{..., cases|items: [...]}], missing}.
Run it right after every wave. Until then the results exist only in the workflow's return value.
"""
import json
import os
import sys

if len(sys.argv) < 3:
    sys.exit(__doc__)
raw, store = sys.argv[1], sys.argv[2]
key = sys.argv[sys.argv.index("--key") + 1] if "--key" in sys.argv else "id"
d = json.load(open(raw))
r = d.get("result", d)
if isinstance(r, str):
    r = json.loads(r)
rows = {}
if os.path.exists(store):
    for ln in open(store):
        if ln.strip():
            x = json.loads(ln)
            rows[x[key]] = x
n = 0
for res in r.get("results", []):
    for x in res.get("cases", res.get("items", [])):
        x["_from"] = os.path.basename(res.get("chunk") or res.get("batch") or "")
        rows[x[key]] = x
        n += 1
tmp = store + ".tmp"
with open(tmp, "w") as fh:
    for x in rows.values():
        fh.write(json.dumps(x) + "\n")
os.replace(tmp, store)
rej = [x for res in r.get("results", []) for x in res.get("rejections", [])]
if rej:  # review workflow: lessons from PRs closed without merging
    rpath = store.rsplit(".jsonl", 1)[0] + "-rejections.jsonl"
    with open(rpath, "a") as fh:
        for x in rej:
            fh.write(json.dumps(x) + "\n")
    print(f"{len(rej)} rejection lessons appended to {rpath}")
miss = r.get("missing", [])
print(
    f"{n} rows upserted into {store} ({len(rows)} total); {len(miss)} inputs missing"
    + (": rerun them" if miss else "")
)
for m in miss:
    print("  MISSING", m)
