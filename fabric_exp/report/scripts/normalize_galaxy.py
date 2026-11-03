#!/usr/bin/env python3
"""Make the Galaxy files follow the P150 convention: value = mean of the passes (rounded to 0.1) where a record lists
two or more passes; the first pass is kept as value_pass1. Idempotent (records with value_pass1 are left alone)."""
import json, os, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(os.path.dirname(HERE), "data")

for name in ("bandwidth_galaxy.json", "galaxy_v4_only.json"):
    p = os.path.join(D, name)
    d = json.load(open(p))
    n = 0
    for r in d:
        ps = r.get("passes") or []
        if len(ps) >= 2 and "value_pass1" not in r and isinstance(r.get("value"), (int, float)):
            r["value_pass1"] = r["value"]
            r["value"] = round(statistics.mean(ps), 1)
            r["notes"] = (r.get("notes") or "") + "; value = mean of the passes (value_pass1 = pass 1)"
            n += 1
    json.dump(d, open(p, "w"), indent=1)
    print(name, n, "records set to the pass mean")
