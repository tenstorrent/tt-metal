#!/usr/bin/env python3
"""slots_table.py PREFIX SUITE [TAG ...]: cycles/packet of every row of fabric_exp/results/PREFIX_TAG_SUITE.csv, one
column per TAG (default: every tag found, `def` first), with each column's change against `def` per row and the
mean change. Cycles/packet = 1.35e9 / packets per second. The depths each tag got are read from its log
(TT_FABRIC_EXP_LOG_SLOTS)."""
import csv, glob, os, re, statistics, sys

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
prefix, suite, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
if not tags:
    found = [os.path.basename(f)[len(prefix) + 1 : -len(suite) - 5] for f in glob.glob(f"{R}/{prefix}_*_{suite}.csv")]
    tags = ["def"] + sorted(t for t in found if t != "def" and "_val" not in t)


def rows(tag):
    f = f"{R}/{prefix}_{tag}_{suite}.csv"
    if not os.path.exists(f):
        return {}
    out = {}
    for r in csv.DictReader(open(f)):
        k = f'{r["test_name"]} {r["ntype"].replace("NOC_UNICAST_", "").lower()} {r["packet_size"]}'
        out[k] = 1.35e9 / float(r["avg_packets_per_s"])
    return out


def depths(tag):
    f = f"{R}/{prefix}_{tag}_{suite}.log"
    if not os.path.exists(f):
        return "?"
    for line in open(f, errors="replace"):
        m = re.search(r"VC0 sender slots \{([^}]*)\} receiver slots \{([^}]*)\}", line)
        if m and not re.search(r"\b0\b", m.group(1)):
            return f"{m.group(1).replace(' ', '')}/{m.group(2)}"
    return "?"


data = {t: rows(t) for t in tags}
base = data.get("def", {})
names = list(base) or sorted({k for d in data.values() for k in d})
w = max(len(n) for n in names)
print(f"{'':{w}}  " + "  ".join(f"{t:>14}" for t in tags))
print(f"{'senders/receiver':{w}}  " + "  ".join(f"{depths(t):>14}" for t in tags))
for n in names:
    cells = []
    for t in tags:
        v = data[t].get(n)
        if v is None:
            cells.append(f"{'-':>14}")
        elif t == "def" or n not in base:
            cells.append(f"{v:>14.1f}")
        else:
            cells.append(f"{v:>7.1f} {v - base[n]:>+6.1f}")
    print(f"{n:{w}}  " + "  ".join(cells))
means = []
for t in tags:
    d = [data[t][n] - base[n] for n in names if n in data[t] and n in base]
    means.append(f"{statistics.mean(d):>+14.2f}" if d and t != "def" else f"{'':>14}")
print(f"{'mean change':{w}}  " + "  ".join(means))
