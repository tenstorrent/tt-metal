"""Per-core breakdown of the SDPA attention phase after the halo release.
Usage: python parse_attn.py <profile_log_device.csv> <calls>"""

import collections
import csv
import statistics as st
import sys

path, calls = sys.argv[1], int(sys.argv[2])
MHZ = 1350.0
open_starts = {}
zones = collections.defaultdict(lambda: collections.defaultdict(list))  # (dev,run) -> (core) -> [(zone,s,e,risc)]
kend = collections.defaultdict(dict)  # (dev,run) -> core -> last kernel end
kstart = collections.defaultdict(lambda: float("inf"))
with open(path) as f:
    next(f)
    r = csv.reader(f)
    next(r)
    for x in r:
        dev, cx, cy, risc, t, run, z, k = (
            x[0].strip(),
            x[1],
            x[2],
            x[3],
            int(x[5]),
            x[7].strip(),
            x[10].strip(),
            x[11].strip(),
        )
        key, core = (dev, run), (int(cx), int(cy))
        if z.endswith("-KERNEL"):
            if k == "ZONE_START":
                kstart[key] = min(kstart[key], t)
            elif k == "ZONE_END":
                kend[key][core] = max(kend[key].get(core, 0), t)
            continue
        if z in ("SDPA_HALO_WAIT", "R_KCHUNK", "R_Q", "C_MERGE"):
            sk = (key, core, risc, z)
            if k == "ZONE_START":
                open_starts.setdefault(sk, []).append(t)
            elif k == "ZONE_END" and open_starts.get(sk):
                zones[key][core].append((z, open_starts[sk].pop(), t, risc))

runs = collections.defaultdict(list)
for d, rn in kstart:
    runs[d].append(int(rn))
keep = [(d, str(rn)) for d, rs in runs.items() for rn in sorted(rs)[-calls:]]

per_call = collections.defaultdict(list)
for key in keep:
    t0 = kstart[key]
    us = lambda c: (c - t0) / MHZ
    prog_end = max(kend[key].values())
    cores = {c: zs for c, zs in zones[key].items() if any(z == "R_KCHUNK" for z, *_ in zs)}
    releases, post_chunks, chunk_durs, q_after, merges, busy, idle_tail = [], [], [], [], [], [], []
    for core, zs in cores.items():
        wait = [e for z, s, e, _ in zs if z == "SDPA_HALO_WAIT"]
        rel = max(wait) if wait else t0
        releases.append(us(rel))
        ks = sorted((s, e) for z, s, e, _ in zs if z == "R_KCHUNK")
        after = [(max(s, rel), e) for s, e in ks if e > rel]
        post_chunks.append(len(after))
        chunk_durs += [(e - s) / MHZ for s, e in after]
        q_after += [(e - s) / MHZ for z, s, e, _ in zs if z == "R_Q" and e > rel]
        m = [(e - s) / MHZ for z, s, e, _ in zs if z == "C_MERGE"]
        if m:
            merges.append(max(m))
        busy.append((kend[key][core] - rel) / MHZ)
    rel_max = max(releases)
    per_call["SDPA cores with K work"].append(len(cores))
    per_call["release (last core) µs"].append(rel_max)
    per_call["K chunks per core after release (median)"].append(st.median(post_chunks))
    per_call["K chunks per core after release (max)"].append(max(post_chunks))
    per_call["time per K chunk after release (median) µs"].append(st.median(chunk_durs))
    if q_after:
        per_call["Q read after release (median) µs"].append(st.median(q_after))
    if merges:
        per_call["K-split merge (median core) µs"].append(st.median(merges))
    per_call["core busy after release (median) µs"].append(st.median(busy))
    per_call["core busy after release (max) µs"].append(max(busy))
    per_call["program end - last release µs"].append((prog_end - t0) / MHZ - rel_max)

print(f"{len(keep)} (chip, call) samples")
for name, v in per_call.items():
    print(f"{name:46s} median {st.median(v):7.1f}   min {min(v):7.1f}   max {max(v):7.1f}")
