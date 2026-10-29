#!/usr/bin/env python3
"""se_table.py [MODE ...]: main's router per single-ERISC mode (se_runs.sh): cycles/packet (1.35 GHz / packets per
second) of the 2D 4352 B, 1D and VC1 split rows, the first mode in full and the others as differences to it, then per
suite the mean difference. Default modes: two fab1 aer1."""
import csv, os, statistics, sys

R = "/home/aagarwal/tt-metal/fabric_exp/results"


def rows(mode, suite):
    if suite == "vc1":
        f, col = f"{R}/se_{mode}vc1_v1_default_p1_auto_13129088_vc1_tests.csv", "packets_per_second"
    else:
        f, col = f"{R}/se_{mode}_v1_13129088_{suite}.csv", "avg_packets_per_s"
    out = {}
    if not os.path.exists(f):
        return out
    for r in csv.DictReader(open(f)):
        name = r["test_name"].replace("_ntype_unicast_write", "")
        k = f'{name} {r.get("ntype", "").replace("NOC_UNICAST_", "").lower()} {r["packet_size"]}'.replace("  ", " ")
        while k in out:
            k += "'"
        out[k] = 1.35e9 / float(r[col])
    return out


def main():
    modes = sys.argv[1:] or ["two", "fab1", "aer1"]
    ref = modes[0]
    for suite in ("4352", "1d", "vc1"):
        data = {m: rows(m, suite) for m in modes}
        names = list(data[ref])
        if not names:
            continue
        w = max(len(n) for n in names)
        print(f"== {suite}\n{'row':{w}}  " + "  ".join(f"{m:>8}" for m in modes))
        for n in names:
            cells = [f"{data[ref][n]:8.1f}"]
            for m in modes[1:]:
                cells.append(f"{data[m][n] - data[ref][n]:+8.1f}" if n in data[m] else f"{'-':>8}")
            print(f"{n:{w}}  " + "  ".join(cells))
        means = [f"{m} {statistics.mean(data[m][n] - data[ref][n] for n in names if n in data[m]):+.2f}"
                 for m in modes[1:] if any(n in data[m] for n in names)]
        print(f"{'mean vs ' + ref:{w}}  " + ", ".join(means) + "\n")


if __name__ == "__main__":
    main()
