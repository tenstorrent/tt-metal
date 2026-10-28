#!/usr/bin/env python3
"""vca_table.py PREFIX[,PREFIX2] ALLOC... : cycles/packet (1.35 GHz / packets per second) of the VC1 split benchmark rows, one
column per allocation, from results/<PREFIX>_<alloc>_p<pass>_o3_13129088_vc1_tests.csv. Each cell is the mean over the
passes found, with the pass-to-pass spread in brackets when there are two. A test with several sending links (the
merge rows) has one row per link (device, direction). The first allocation is the reference: the other columns show
their difference to it. VCA_OPT=auto reads labels of runs at the builder's level. With PREFIX2 (e.g. vcb_v4gi,vcb_v4git) each cell is instead PREFIX2 - PREFIX at that
allocation (same job, interleaved runs), after the plain table of PREFIX."""
import csv, os, sys
from collections import OrderedDict

RES = "/home/aagarwal/tt-metal/fabric_exp/results"
OPT = os.environ.get("VCA_OPT", "o3")  # the label's level suffix (main's router runs at the builder's level: auto)


def rows(prefix, alloc, p):
    f = os.path.join(RES, f"{prefix}_{alloc}_p{p}_{OPT}_13129088_vc1_tests.csv")
    if not os.path.exists(f):
        return None
    out = OrderedDict()
    for r in csv.DictReader(open(f)):
        key = r["test_name"]
        if key in out or any(k.startswith(key + " @") for k in out):
            if key in out:  # a second link of the same test: key every link by device and direction
                first = out.pop(key)
                out[f"{key} @{first[1]}"] = first
            key = f"{key} @{r['device']}{r['direction']}"
        out[key] = (1.35e9 / float(r["packets_per_second"]), f"{r['device']}{r['direction']}")
    return out


def load(prefix, allocs):
    data = {}
    for a in allocs:
        passes = [x for x in (rows(prefix, a, p) for p in (1, 2)) if x]
        cell = {}
        for k in passes[0] if passes else []:
            v = [p[k][0] for p in passes if k in p]
            cell[k] = (sum(v) / len(v), max(v) - min(v) if len(v) > 1 else None)
        data[a] = cell
    return data


def main():
    prefixes, allocs = sys.argv[1].split(","), sys.argv[2:]
    prefix = prefixes[0]
    if len(prefixes) == 2:
        sys.argv[1] = prefix
        main()
        a_, b_ = load(prefixes[0], allocs), load(prefixes[1], allocs)
        names = list(a_[allocs[0]])
        w = max(len(n) for n in names)
        print(f"\n{prefixes[1]} - {prefixes[0]}")
        print(f"{'row':{w}}  " + "  ".join(f"{a:>9}" for a in allocs))
        for n in names:
            print(f"{n:{w}}  " + "  ".join(
                f"{b_[a][n][0] - a_[a][n][0]:+9.1f}" if n in a_[a] and n in b_[a] else f"{'-':>9}" for a in allocs))
        return
    data = load(prefix, allocs)
    ref = allocs[0]
    names = list(data[ref])
    w = max(len(n) for n in names)
    print(f"{'row':{w}}  " + "  ".join(f"{a:>16}" for a in allocs))
    for n in names:
        line = f"{n:{w}}  "
        for a in allocs:
            if n not in data[a]:
                line += f"{'-':>16}  "
                continue
            m, s = data[a][n]
            spread = f"[{s:.1f}]" if s is not None else ""
            if a == ref:
                line += f"{m:9.1f}{spread:>7}  "
            else:
                line += f"{m - data[ref][n][0]:+9.1f}{spread:>7}  "
        print(line)


if __name__ == "__main__":
    main()
