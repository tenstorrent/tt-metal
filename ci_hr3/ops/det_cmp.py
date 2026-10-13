#!/usr/bin/env python3
"""test_det records: per case, in-process repeat differences per run and the hashes of every variant against the base's.
usage: det_cmp.py <dir with det_<i>_<variant>.jsonl> <base variant>"""
import glob, json, os, sys
d, base = sys.argv[1], sys.argv[2]
per = {}
for f in sorted(glob.glob(os.path.join(d, "det_*.jsonl"))):
    v = os.path.basename(f)[:-6].split("_", 2)[2]
    for l in open(f):
        r = json.loads(l)
        e = per.setdefault(r["case"], {}).setdefault(v, {"h": set(), "diffs": []})
        e["h"].update(r["hashes"]); e["diffs"] += r["in_process_diffs"]
for c in sorted(per):
    vs = per[c]
    b = vs.get(base, {"h": set()})
    parts = [f"{v}: {len(x['h'])} distinct, in-process diffs max {max(x['diffs'] or [0])}, same as {base} {x['h'] == b['h']}" for v, x in sorted(vs.items())]
    print(c, "|", "; ".join(parts))
