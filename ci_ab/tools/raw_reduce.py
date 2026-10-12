"""Splits each device run's KERNEL L1_TO_L1 span (pack KERNEL end - unpack KERNEL start) into its parts, from the raw
zone entries the patched profiler appends (RAW_ZONE_DIR). Per test node, medians over the device runs:
  K   pack KERNEL end - unpack KERNEL start (the KERNEL L1_TO_L1 row)
  a   unpack INIT start - unpack KERNEL start (entry rendezvous of INIT)
  b   pack INIT end - unpack INIT start (the INIT L1_TO_L1 row)
  c   unpack TILE_LOOP start - pack INIT end (INIT exit rendezvous with the counter read, TILE_LOOP entry rendezvous)
  d   pack TILE_LOOP end - unpack TILE_LOOP start (the TILE_LOOP L1_TO_L1 row)
  e   pack KERNEL end - pack TILE_LOOP end (TILE_LOOP exit rendezvous with the counter read, tensix_sync)
  p0  pack INIT start - pack KERNEL start, s0 pack KERNEL start - unpack KERNEL start
usage: raw_reduce.py <run tag> <variant> <raw dir>"""
import glob
import statistics
import sys

import pandas as pd

tag, var, d = sys.argv[1:4]
fs = glob.glob(f"{d}/raw_*.csv")
if not fs:
    print("RZ NORAW", tag, var)
    sys.exit(0)
df = pd.concat(pd.read_csv(f) for f in fs)
out = {}
for (node, seq), g in df.groupby(["node", "seq"]):
    t = {}
    for r in g.itertuples(index=False):
        t[(r.thread, r.marker, r.type)] = r.timestamp
    try:
        uk, ui, ut = t[("unpack", "KERNEL", "ZONE_START")], t[("unpack", "INIT", "ZONE_START")], t[("unpack", "TILE_LOOP", "ZONE_START")]
        pk, pi = t[("pack", "KERNEL", "ZONE_START")], t[("pack", "INIT", "ZONE_START")]
        pie, pte, pke = t[("pack", "INIT", "ZONE_END")], t[("pack", "TILE_LOOP", "ZONE_END")], t[("pack", "KERNEL", "ZONE_END")]
    except KeyError:
        continue
    row = dict(K=pke - uk, a=ui - uk, b=pie - ui, c=ut - pie, d=pte - ut, e=pke - pte, p0=pi - pk, s0=pk - uk)
    out.setdefault(node.split("[", 1)[-1].rstrip("]"), []).append(row)
for node in sorted(out):
    rows = out[node]
    med = {k: statistics.median(r[k] for r in rows) for k in rows[0]}
    print("RZ", tag, var, node.replace(" ", ""), f"n={len(rows)}", " ".join(f"{k}={med[k]:g}" for k in med))
