#!/usr/bin/env python3
"""covcheck.py BUILD [BASE=v4rx1_O3]: coverage results of BUILD: configs built, failures per case (with the first
error), and whether every config BASE built also built here."""
import collections, json, os, sys
COV = os.environ.get("TT_COV_DIR", "/home/aagarwal/.tt-metal-v3r-builds/cov_m") + "/builds"
b = sys.argv[1]
base = sys.argv[2] if len(sys.argv) > 2 else "v4rx1_O3"
r = json.load(open(f"{COV}/{b}/results.json"))
rb = json.load(open(f"{COV}/{base}/results.json"))
fail = collections.defaultdict(list)
for k, v in r.items():
    if not v["ok"]:
        fail[v["case"]].append(v["err"])
print(f"{b}: {sum(v['ok'] for v in r.values())}/{len(r)} built")
for c in sorted(fail):
    print(f"  {c}: {len(fail[c])} failed; first: {(fail[c][0] or '')[:200]}")
missing = [k for k, v in rb.items() if v["ok"] and not r.get(k, {}).get("ok")]
print(f"of the {sum(v['ok'] for v in rb.values())} configs {base} built: {len(missing)} not built here")
for k in missing[:10]:
    print("   ", k, (r.get(k, {}).get("err") or "absent")[:200])
