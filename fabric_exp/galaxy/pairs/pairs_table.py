#!/usr/bin/env python3
"""pairs_table.py PAIRS_RAW QUADS_TSV SINFO_TXT: the Galaxy pair table (glxq_pairs.tsv body) from fsd_pairs.py's list:
every bh-glx-110-a host pair with Ethernet connections, its kind (ring neighbours of a quad: 16 connections; a single
cross-quad cable: 2), the Slurm spec, its leaf, the pair MGDs it supports, and the hosts' state (standing excludes,
Slurm drained/down when read)."""
import sys, re, collections
raw, quads_tsv, sinfo = sys.argv[1:4]
EXCLUDE = {"a09u02", "a03u14", "a08u20", "a01u14", "a02u02", "a02u08", "a02u14", "a10u02"}
quad_of, ring_of, leaf_of, part_of = {}, {}, {}, {}
for line in open(quads_tsv):
    if line.startswith("#") or not line.strip():
        continue
    name, slurm, ring, leaf = line.rstrip("\n").split("\t")
    if not name.startswith(("pod", "a0", "a1", "c120_pod")):
        continue
    hosts = ring.split(",")
    for h in hosts:
        quad_of[h], leaf_of[h] = name, leaf
        part_of[h] = ("bh_110a_sc16" if slurm.startswith("bh_110a_sc16") else
                      "bh_sc36_2" if slurm.startswith("bh_sc36_2") else "bh_sc36_6")
    ring_of[name] = hosts
state = collections.defaultdict(set)
for line in open(sinfo):
    f = line.split()
    if len(f) >= 3:
        state[f[0]].add(f[2])
short = lambda h: h.replace("bh-glx-110-", "").replace("bh-glx-120-", "")
rows = []
for line in open(raw):
    a, b, n = line.split()[:3]
    n = int(n)
    if a not in quad_of or b not in quad_of:
        continue
    qa, qb = quad_of[a], quad_of[b]
    if qa == qb:
        r = ring_of[qa]
        i, j = r.index(a), r.index(b)
        assert (j - i) % 4 in (1, 3), (a, b)
        if (j - i) % 4 == 3:
            a, b = b, a  # ring order
        kind, name = "ring", f"{qa}_{short(a)}_{short(b)}"
    else:
        kind, name = "cable", f"x_{short(a)}_{short(b)}"
    if part_of[a] != part_of[b]:
        slurm = "-"  # crosses partitions: one job cannot take it
    else:
        slurm = f"{part_of[a]}:{a},{b}"
    leaf = leaf_of[a] if leaf_of[a] == leaf_of[b] else f"{leaf_of[a]}|{leaf_of[b]}"
    if n == 16:
        mgds = "16x4,mm_exp,mm_exp8,mm_zfb,mm_zfb8,express16,torusx16"
    else:
        mgds = "-"
    bad = sorted({short(h) for h in (a, b) if short(h) in EXCLUDE})
    sl = sorted({f"{short(h)}:{s}" for h in (a, b) for s in state.get(h, {"?"}) if s not in ("idle", "allocated", "mixed", "completing")})
    status = ";".join((["exclude:" + ",".join(bad)] if bad else []) + sl) or "ok"
    rows.append((kind != "ring", name, slurm, f"{a},{b}", leaf, str(n), mgds, status))
for r in sorted(rows):
    print("\t".join(r[1:]))
