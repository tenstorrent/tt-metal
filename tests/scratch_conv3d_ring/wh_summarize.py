#!/usr/bin/env python3
# SCRATCH (conv3d ring experiments). usage: wh_summarize.py <dir> <basename...>
# Host wall-clock summary of prod_ab.py runs: each <basename>.meta.json records, per entry, the wall time of
# warmup + timed calls between two device synchronizes. Prints one JSON document to stdout.
import json
import os
import sys

d, names = sys.argv[1], sys.argv[2:]
res, grid = {}, None
for base in names:
    mp = f"{d}/{base}.meta.json"
    if not os.path.exists(mp):
        print(f"WARN missing {base}", file=sys.stderr)
        continue
    arm = base.split("_prod_")[1]
    meta = json.load(open(mp))
    grid = meta.get("grid")
    k = meta["warmup"] + meta["n_timed"]
    for e in meta["entries"]:
        if e["status"] != "ok" or tuple(e["kernel"]) == (1, 1, 1):
            continue
        rec = res.setdefault(
            e["name"], {"meta": {kk: v for kk, v in e.items() if kk not in ("status", "wall_s", "checksum")}}
        )
        rec.setdefault(arm, []).append(round(e["wall_s"] * 1e6 / k, 2))  # host wall us per call, warmup included
        rec["meta"].setdefault("checksum", {})[arm] = e.get("checksum")
print(json.dumps({"grid": grid, "method": "host wall-clock per call, warmup included", "entries": res}, sort_keys=True))
