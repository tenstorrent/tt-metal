#!/usr/bin/env python3
"""cisp_slowpos.py ON_CSV OFF_CSV [ON_CSV OFF_CSV ...]: slow iterations (row below 97% of the off median) by position
inside the 18-iteration half-repeat of the NX yaml (0-5: the unicast_write block), blocks hit per run, and the ratio
spread per ntype."""
import csv, statistics, collections, sys, re
def load(p): return list(csv.DictReader(open(p)))
a = sys.argv[1:]
pos = collections.Counter(); blocks = collections.Counter(); rat = collections.defaultdict(list)
for i in range(0, len(a), 2):
    offm = collections.defaultdict(list)
    for r in load(a[i+1]): offm[r["test_name"]].append(float(r["bandwidth_GB_s"]))
    rows = load(a[i]); hit = set()
    for j, r in enumerate(rows):
        x = float(r["bandwidth_GB_s"]) / statistics.median(offm[r["test_name"]])
        nt = re.search(r"ntype_(\w+?)_num_links", r["test_name"]).group(1)
        rat[nt].append(x)
        if x < 0.97: pos[j % 18] += 1; hit.add(j // 18)
    blocks[len(hit)] += 1
print("slow by position within the 18-iteration half-repeat (0-5 unicast_write):", dict(sorted(pos.items())))
print("blocks hit per run:", dict(blocks))
for nt, v in rat.items():
    v = sorted(v); print(nt, "min %.4f p5 %.4f median %.4f; below 0.99: %d, 0.97-0.99: %d" % (v[0], v[len(v)//20], statistics.median(v), sum(x < 0.99 for x in v), sum(0.97 <= x < 0.99 for x in v)))
