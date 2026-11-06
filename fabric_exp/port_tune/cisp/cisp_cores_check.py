#!/usr/bin/env python3
"""cisp_cores_check.py ZONES_DIR: do the NX worker cores (senders, receivers, sync) of a row change with what ran before
it? Offline, from the 823 zone CSV (profiler on, stock CI order: per size, write / fused / scatter, each L1 x3 then
L2 x3, so every L1 block but the run's first follows an L2 block).
Per (chip, iteration): every core with a BRISC-KERNEL start within 1.5 ms of that chip's sync-core (15,11) kernel start
(raw per-chip cycles). The cores of each row type (size, ntype, num_links) are compared across all its iterations,
the run's first L1 block (no L2 before it in the process) against the later ones (after an L2 block)."""
import bisect, collections, csv, os, re, sys

D = sys.argv[1]
z = collections.defaultdict(list)
with open(os.path.join(D, "zones_on_zones.csv")) as f:
    next(f); next(f)
    for line in f:
        p = [x.strip() for x in line.split(",")]
        if len(p) > 11 and p[10] == "BRISC-KERNEL" and p[11] == "ZONE_START":
            z[(int(p[0]), int(p[1]), int(p[2]))].append(int(p[5]))
chips = sorted({k[0] for k in z})
rows = list(csv.DictReader(open(os.path.join(D, "zones_on_bandwidth_results.csv"))))
KEY = re.compile(r"size_(\d+)_ntype_(\w+?)_num_links_(\d)$")
cores = collections.defaultdict(set)  # (chip, j) -> {(x, y)}
for c in chips:
    starts = sorted(z[(c, 15, 11)])
    assert len(starts) == len(rows), (c, len(starts), len(rows))
    for (cc, x, y), ss in z.items():
        if cc != c:
            continue
        for s in ss:
            j = bisect.bisect_left(starts, s)
            j = min((q for q in (j - 1, j) if 0 <= q < len(starts)), key=lambda q: abs(starts[q] - s))
            if abs(starts[j] - s) <= 2_000_000:
                cores[(c, j)].add((x, y))
key = [KEY.search(r["test_name"]).groups() for r in rows]
after_l2 = [j > 0 and any(key[q][2] == "2" for q in range(j)) for j in range(len(rows))]
print(f"{len(rows)} iterations, chips {chips}; the first L1 block (no L2 before it): "
      f"{[j for j in range(len(rows)) if key[j][2] == '1' and not after_l2[j]]}")
same_all = True
for k in sorted(set(key)):
    js = [j for j in range(len(rows)) if key[j] == k]
    for c in chips:
        sets = collections.Counter(frozenset(cores[(c, j)]) for j in js)
        if len(sets) != 1:
            same_all = False
            print(f"   DIFFER: {k} chip {c}: {len(sets)} core sets over {len(js)} iterations")
    first = [j for j in js if not after_l2[j]]
    print(f"   {k[1]:22s} {k[0]} B, {k[2]} link(s): {len(js)} iterations ({len(first)} before any L2); per chip one core set: "
          f"{all(len({frozenset(cores[(c, j)]) for j in js}) == 1 for c in chips)}")
print("every row type's cores identical in all its iterations, on every chip:", same_all)
for k in (("2048", "unicast_write", "1"), ("2048", "unicast_write", "2")):
    j = next(j for j in range(len(rows)) if key[j] == k)
    print(f"   cores of {k} (iteration {j}):")
    for c in chips:
        s = sorted(cores[(c, j)])
        print(f"      chip {c}: senders/sync x=15 {[xy for xy in s if xy[0] == 15]}, others {[xy for xy in s if xy[0] != 15]}")
