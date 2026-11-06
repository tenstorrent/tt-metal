#!/usr/bin/env python3
"""cisp_order_report.py RESULTS_DIR: the row-order checks of cisp_order_ab.sh. Per variant (S stock, A write only, B
write block last, C one write per row with a tiny fused row after each): slow rows (the on arm's row below 97% of the
off arm's median for the same size / ntype / num_links, over all repeats), slow counts per ntype, the pattern per
repeat for both arms (S slow, . normal, g a gap row), and slow write iterations by position inside their write block."""
import collections, csv, os, re, statistics, sys

D = sys.argv[1]
KEY = re.compile(r"size_(\d+)_ntype_(\w+?)_num_links_(\d)$")
REPEAT = {"S": 36, "A": 12, "B": 36, "C": 24}


def rows(v, arm):
    p = os.path.join(D, f"ord_{v}_{arm}_bandwidth_results.csv")
    if not os.path.exists(p):
        return None
    out = []
    for r in csv.DictReader(open(p)):
        m = KEY.search(r["test_name"])
        out.append((m.group(0), m.group(2), float(r["bandwidth_GB_s"])))
    return out


def print_arm(v, arm, rr, med):
    flags, by, tot, rat = [], collections.Counter(), collections.Counter(), collections.defaultdict(list)
    on = rr
    for k, nt, bw in on:
        x = bw / med[k]
        rat[nt].append(x)
        tot[nt] += 1
        s = x < 0.97
        by[nt] += s
        gap = v == "C" and nt != "unicast_write"
        flags.append("g" if gap and not s else ("G" if gap else ("S" if s else ".")))
    n = REPEAT[v]
    print(f"\n== {v}, arm {arm} (against the off median): {len(on)} rows; slow by ntype: "
          + ", ".join(f"{t} {by[t]}/{tot[t]} (min {min(rat[t]):.4f})" for t in tot))
    s = "".join(flags)
    print("   " + "\n   ".join(s[i:i + n] for i in range(0, len(s), n)))
    # slow write iterations by position inside their write block (a maximal run of consecutive write rows; in C the
    # writes of one repeat, gaps skipped)
    pos = collections.Counter()
    nblk = hit = 0
    if v == "C":
        blocks = [[j for j in range(i, min(i + n, len(on))) if on[j][1] == "unicast_write"] for i in range(0, len(on), n)]
    else:
        blocks, cur = [], []
        for j, (_, nt, _) in enumerate(on):
            if nt == "unicast_write" and (not cur or j == cur[-1] + 1) and (v != "A" or j % n):
                cur.append(j)
            elif nt == "unicast_write":
                if cur:
                    blocks.append(cur)
                cur = [j]
            else:
                if cur:
                    blocks.append(cur)
                cur = []
        if cur:
            blocks.append(cur)
    for b in blocks:
        nblk += 1
        hit += any(flags[j] == "S" for j in b)
        for p, j in enumerate(b):
            pos[p] += flags[j] == "S"
    L = max(len(b) for b in blocks)
    print(f"   write blocks {nblk} (length {L}), {hit} with a slow iteration; slow by position in block: "
          + " ".join(f"{p}:{pos[p]}" for p in range(L)))


for v in ("S", "A", "B", "C"):
    on, off = rows(v, "on"), rows(v, "off")
    if not on or not off:
        print(f"== {v}: missing CSVs")
        continue
    med = collections.defaultdict(list)
    for k, _, bw in off:
        med[k].append(bw)
    med = {k: statistics.median(x) for k, x in med.items()}
    offr = [bw / med[k] for k, _, bw in off]
    for arm, rr in (("off", off), ("on", on)):
        print_arm(v, arm, rr, med)

