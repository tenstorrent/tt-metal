#!/usr/bin/env python3
"""Scratch, not for merge. rec.py <jsonl> <unit> <run> <arm>: one log line per test, device kernel ns summed over its programs."""
import json, sys
f, u, r, a = sys.argv[1:5]
try:
    L = open(f).read().splitlines()
except OSError:
    L = []
for l in L:
    x = json.loads(l); p = x.get("progs") or {}
    ns = sum(sum(v) for v in p.values()) if (x.get("passed") and p) else "NA"
    print(f"CIAB R {u} {r} {a} {ns} {int(bool(x.get('passed')))} {(x.get('bits') or '-')[:8]} {x['id']}")
print(f"CIAB N {u} {r} {a} records {len(L)}")
