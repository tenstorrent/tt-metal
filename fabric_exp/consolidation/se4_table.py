#!/usr/bin/env python3
"""se4_table.py [TWO_LABEL]: v4se per ERISC mode against main's router per mode, cycles/packet (1.35 GHz / packets per
second) of the 2D 4352 B, 1D and VC1 split rows, per row and then per suite the mean single-ERISC cost (mode - two) of
each router. v4se single-ERISC: s1<mode>_v4se_13129088_<suite> (se4_runs.sh); v4se two-ERISC: TWO_LABEL (default s1two_v4gi, job 121,
merge_ab_runs.sh); main: se_<mode>_v1_13129088_<suite> (se_runs.sh). VC1: s1<mode>vc1_v4se_default_p1_o3 in every mode
(se4vc1_runs.sh), main se_<mode>vc1_v1_default_p1_auto. Rows the two routers do not share are left out."""
import csv, os, statistics, sys

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
MODES = ("two", "aer1", "fab1")
SUITES = (("4352", "4352", "4352"), ("1d", "1d_m1", "1d"), ("vc1", None, None))  # name, v4 suite, main suite


def rows(f):
    out = {}
    if not os.path.exists(f):
        return out
    for r in csv.DictReader(open(f)):
        if "avg_packets_per_s" in r:
            k, pps = f'{r["test_name"]} {r["ntype"].replace("NOC_UNICAST_", "").lower()}', r["avg_packets_per_s"]
        else:  # VC1 split: one row per sender, the test name carries size and send type
            k, pps = r["test_name"].replace("_ntype_unicast_write", ""), r["packets_per_second"]
        while k in out:
            k += "'"
        out[k] = 1.35e9 / float(pps)
    return out


def v4_file(two, mode, suite):
    if suite is None:
        return f"{R}/s1{mode}vc1_v4se_default_p1_o3_13129088_vc1_tests.csv"
    return f"{R}/{two}_13129088_{suite}.csv" if mode == "two" else f"{R}/s1{mode}_v4se_13129088_{suite}.csv"


def main_file(mode, suite):
    if suite is None:
        return f"{R}/se_{mode}vc1_v1_default_p1_auto_13129088_vc1_tests.csv"
    return f"{R}/se_{mode}_v1_13129088_{suite}.csv"


def main():
    two = sys.argv[1] if len(sys.argv) > 1 else "s1two_v4gi"
    for name, v4s, mains in SUITES:
        v4 = {m: rows(v4_file(two, m, v4s)) for m in MODES}
        mn = {m: rows(main_file(m, mains)) for m in MODES}
        keys = [k for k in mn["two"] if k in v4["two"]]
        print(f"== {name} cycles/packet: " + "  ".join(f"v4 {m:>4}" for m in MODES) + "   " +
              "  ".join(f"main {m:>4}" for m in MODES))
        for k in keys:
            cell = lambda d: f"{d[k]:9.0f}" if k in d else f"{'-':>9}"
            print(f"  {k:42}" + " ".join(cell(v4[m]) for m in MODES) + "   " + " ".join(f" {cell(mn[m])}" for m in MODES))
        for m in MODES[1:]:
            for router, d in (("v4", v4), ("main", mn)):
                ks = [k for k in keys if k in d[m] and k in d["two"]]
                if ks:
                    deltas = [d[m][k] - d["two"][k] for k in ks]
                    print(f"  {router:4} {m} - two: mean {statistics.mean(deltas):+6.0f}  min {min(deltas):+5.0f}"
                          f"  max {max(deltas):+5.0f}  ({len(ks)} rows)")
            ks = [k for k in keys if k in v4[m] and k in mn[m]]
            if ks:
                deltas = [v4[m][k] - mn[m][k] for k in ks]
                print(f"  v4 - main in {m}: mean {statistics.mean(deltas):+6.0f}  min {min(deltas):+5.0f}"
                      f"  max {max(deltas):+5.0f}  ({len(ks)} rows)")


if __name__ == "__main__":
    main()
