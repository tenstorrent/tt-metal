#!/usr/bin/env python3
"""census_report.py [--rows] [SUITE ...]: the perf census (census_runs.sh, census_vc1_runs.sh) per suite: cycles/packet
(1.35 GHz) of main as shipped (main_fw), main at the study keepalive (main), v3qm and v4trm, each the mean of passes a
and b, with the suite mean and v4trm against each other router. --rows lists every row; rows whose two passes differ by
more than 3% for a router are marked '~' (noisy). MixCmd rows are left out of means (bimodal). VC1 suites keep the
slowest sender per test."""
import csv, os, statistics, sys

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
ROUTERS = ["main_fw", "main", "v3qm", "v4trm"]
PASSES = ["a", "b"]
REF = "v4trm"  # the column the others are compared with
for arg in sys.argv[1:]:
    if arg.startswith("--routers="):
        ROUTERS = arg.split("=", 1)[1].split(",")
    elif arg.startswith("--passes="):
        PASSES = arg.split("=", 1)[1].split(",")
    elif arg.startswith("--ref="):
        REF = arg.split("=", 1)[1]
SUITES = ["4352", "1d", "mix", "burst", "ring", "torusy", "4352_1e", "1d_1e", "rse", "mux", "trim", "vc1", "vc1i"]


def path(p, router, suite):
    if suite in ("vc1", "vc1i"):
        opt = "auto" if router == "main_fw" else "o3"
        return f"{R}/cen{p}{suite}_{router}_default_p1_{opt}_13129088_vc1_tests.csv"
    return f"{R}/cen{p}_{router}_{suite}.csv"


def rows(f):
    if not os.path.exists(f):
        return None
    out = {}
    for r in csv.DictReader(open(f)):
        if "avg_packets_per_s" in r:
            k = (r["test_name"], r["ntype"].replace("NOC_UNICAST_", "").lower(), r["packet_size"])
            v = 1.35e9 / float(r["avg_packets_per_s"])
        else:
            k = (r["test_name"], "", "")
            v = 1.35e9 / float(r["packets_per_second"])
            if k + (0,) in out:
                out[k + (0,)] = max(out[k + (0,)], v)
                continue
        n = sum(1 for kk in out if kk[:3] == k)
        out[k + (n,)] = v
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show_rows = "--rows" in sys.argv
    summary = []
    for s in args or SUITES:
        data = {}
        for r in ROUTERS:
            ds = [rows(path(p, r, s)) for p in PASSES]
            if all(d is None for d in ds):
                continue
            ks = set().union(*[set(d.keys()) for d in ds if d])
            data[r] = {k: [d[k] for d in ds if d and k in d] for k in ks}
        if not data:
            continue
        cols = [r for r in ROUTERS if r in data]
        ks = sorted(k for k in data[cols[0]] if all(k in data[c] for c in cols))
        ks_mean = [k for k in ks if "MixCmd" not in k[0]]
        print(f"\n## {s} ({len(ks_mean)} rows; passes: " + ", ".join(f"{c} {max(len(v) for v in data[c].values())}" for c in cols) + ")")
        print(f"{'row':56}" + "".join(f"{c:>11}" for c in cols))
        if show_rows:
            for k in ks:
                lab = f"{k[0]} {k[1]} {k[2]}" + (f" #{k[3]}" if k[3] else "")
                cells = ""
                for c in cols:
                    v = data[c][k]
                    noisy = len(v) == 2 and abs(v[0] - v[1]) > 0.03 * min(v)
                    cells += f"{statistics.mean(v):10.1f}{'~' if noisy else ' '}"
                print(f"{lab[:56]:56}" + cells)
        means = {c: statistics.mean(statistics.mean(data[c][k]) for k in ks_mean) for c in cols}
        print(f"{'MEAN':56}" + "".join(f"{means[c]:10.1f} " for c in cols))
        if REF in means:
            print(f"{REF + ' / router':56}" + "".join(f"{means[REF] / means[c]:10.3f} " for c in cols))
        summary.append((s, len(ks_mean), means))
    print("\n## summary (suite means, cycles/packet; ratios = each router / " + REF + ")")
    print(f"{'suite':10}{'rows':>5}" + "".join(f"{c:>10}" for c in ROUTERS) + "".join(f"{c + '/' + REF:>16}" for c in ROUTERS if c != REF))
    for s, n, m in summary:
        print(f"{s:10}{n:5}" + "".join(f"{m[c]:10.1f}" if c in m else f"{'-':>10}" for c in ROUTERS)
              + "".join(f"{(m[c] / m[REF] if c in m and REF in m else float('nan')):16.3f}" for c in ROUTERS if c != REF))

if __name__ == "__main__":
    main()
