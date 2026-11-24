#!/usr/bin/env python3
"""Sum one session's per-ELF stats (tu.*.jsonl) into a JSON object on stdout."""
import glob
import json
import sys
from collections import Counter

stats_dir = sys.argv[1]
results = Counter()
sums = Counter()
tus = 0
for path in glob.glob(f"{stats_dir}/tu.*.jsonl"):
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            tus += 1
            results[r["result"]] += 1
            for k in ("compile_s", "pre_s", "lookup_s", "store_s", "total_s"):
                sums[k] += r.get(k, 0.0)
depsets = sum(1 for p in glob.glob(f"{stats_dir}/depsets.*.jsonl") for _ in open(p))
print(json.dumps({"tus": tus, "results": dict(results), "seconds": {k: round(v, 1) for k, v in sums.items()},
                  "depset_lines": depsets}))
