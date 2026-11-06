#!/usr/bin/env python3
# SCRATCH (conv3d ring experiments). usage: wh_summarize.py <dir> <basename...>
# Parses the streaming-profiler CSVs written by prod_ab.py (<dir>/<basename>.csv + <basename>.meta.json),
# maps rows to entries with the triple 1-core sentinel, keeps the main program of each timed call, and
# prints one JSON document to stdout.
import csv
import json
import os
import statistics
import sys

d, names = sys.argv[1], sys.argv[2:]
res = {}
for base in names:
    f, mp = f"{d}/{base}.csv", f"{d}/{base}.meta.json"
    if not (os.path.exists(f) and os.path.exists(mp)):
        print(f"WARN missing {base}", file=sys.stderr)
        continue
    arm = base.split("_prod_")[1]
    meta = json.load(open(mp))
    rows = list(csv.DictReader(open(f)))
    k = meta["warmup"] + meta["n_timed"]
    cc = [r["CORE COUNT"] for r in rows]
    segs, start = [], 0
    for i in range(5, len(rows)):
        # sentinel block = [prep, prep, (prelude, 1-core main) x 3]; match on the three 1-core mains
        if (
            cc[i] == "1"
            and cc[i - 2] == "1"
            and cc[i - 4] == "1"
            and cc[i - 1] == cc[i - 3] == cc[i - 5]
            and cc[i - 1] != "1"
        ):
            segs.append(rows[start : i - 7])  # drop the sentinel's 2 prep rows + 6 call rows
            start = i + 1
    if len(segs) != len(meta["entries"]):
        print(f"WARN {base}: {len(segs)} segments vs {len(meta['entries'])} entries", file=sys.stderr)
    for e, seg in zip(meta["entries"], segs):
        if e["status"] != "ok" or len(seg) < 2 * k or tuple(e["kernel"]) == (1, 1, 1):
            continue
        pairs = seg[-2 * k :]  # [entry prep rows..., (prelude, main) x k]
        preludes, mains = pairs[0::2], pairs[1::2]
        cores = {r["CORE COUNT"] for r in mains}
        pcores = {r["CORE COUNT"] for r in preludes}
        if len(cores) != 1 or len(pcores) != 1:
            print(f"WARN {base} {e['name']}: core counts mains {cores} preludes {pcores}", file=sys.stderr)
            continue
        dur = [float(r["DEVICE KERNEL DURATION [ns]"]) / 1000 for r in mains[meta["warmup"] :]]
        rec = res.setdefault(e["name"], {"meta": {kk: v for kk, v in e.items() if kk not in ("status", "wall_s")}})
        rec["meta"]["cores"] = int(next(iter(cores)))
        rec.setdefault(arm, []).append(round(statistics.mean(dur), 3))
        rec["meta"].setdefault("checksum", {})[arm] = e.get("checksum")
print(json.dumps({"grid": meta.get("grid") if names else None, "entries": res}, sort_keys=True))
