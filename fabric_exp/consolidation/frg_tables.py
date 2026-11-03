#!/usr/bin/env python3
"""frg_tables.py: cycles/packet tables of the FULLRING_MCAST_GAP.md A/B runs (1.35e9 / the summary CSV's
avg_packets_per_s, as report/data/bandwidth_glx_sweep.json).

  frg_tables.py glx JOB_RESULTS_DIR      Galaxy (frs_glx_frgap.sbatch): labels PASS_CONFIG_SUITE
  frg_tables.py p150 RESULTS_DIR TAG     P150 (frg_p150_runs.sh): labels frg<PASS>_<router>_p150ring
  frg_tables.py p150it RESULTS_DIR       P150 injection throttle (frg_it_p150_runs.sh): frgit<PASS>_<arm>_<rows>

Prints one row per (suite, test, size, ntype) and one column per config with "pass a / pass b (mean)", and each run's
logged slot allocation (TT_FABRIC_EXP_LOG_SLOTS=1)."""
import collections
import csv
import glob
import os
import re
import sys


def rates(csv_path):
    out = {}
    for r in csv.DictReader(open(csv_path)):
        nt = r.get("ntype", "")
        out[(r["test_name"], int(r["packet_size"]), nt)] = 1.35e9 / float(r["avg_packets_per_s"])
    return out


def slots(log_path):
    if not os.path.exists(log_path):
        return ""
    s = collections.Counter(re.findall(r"VC0 sender slots \{([0-9, ]+)\} receiver slots \{([0-9, ]+)\}",
                                       open(log_path, errors="replace").read()))
    return "; ".join(f"{a}/{b} x{n}" for (a, b), n in s.most_common(2))


def main():
    mode = sys.argv[1]
    runs = collections.defaultdict(dict)  # (suite, config) -> {pass: rates}
    logs = {}
    if mode == "glx":
        d = sys.argv[2]
        for f in sorted(glob.glob(f"{d}/*.csv")):
            b = os.path.basename(f)[:-4]
            if b.endswith("_tests") or "_samples" in b:
                continue
            p, c, s = b.split("_", 2)
            runs[(s, c)][p] = rates(f)
            logs[(s, c, p)] = slots(f[:-4] + ".log")
    elif mode == "p150it":
        d = sys.argv[2]
        for f in sorted(glob.glob(f"{d}/frgit*_*_*.csv")):
            b = os.path.basename(f)[:-4]
            m = re.match(r"frgit([a-uw-z])_(base|A|B|C)_(ring|t24|t42)$", b)
            if not m:
                continue
            runs[(m.group(3), m.group(2))][m.group(1)] = rates(f)
            logs[(m.group(3), m.group(2), m.group(1))] = slots(f[:-4] + ".log")
    else:
        d, tag = sys.argv[2], sys.argv[3]
        for f in sorted(glob.glob(f"{d}/{tag}*_p150ring.csv")):
            b = os.path.basename(f)[:-4]
            m = re.match(rf"{tag}(\w)_(.+)_p150ring$", b)
            if not m:
                continue
            runs[("p150ring", m.group(2))][m.group(1)] = rates(f)
            logs[("p150ring", m.group(2), m.group(1))] = slots(f[:-4] + ".log")
    by_suite = collections.defaultdict(list)
    for (s, c) in runs:
        by_suite[s].append(c)
    for s, cfgs in sorted(by_suite.items()):
        order = sorted(cfgs, key=lambda c: (c != "main", c not in ("v4rxg3", "base"), c))
        keys = sorted({k for c in cfgs for p in runs[(s, c)].values() for k in p})
        print(f"\n## {s}")
        print("| test | size | ntype | " + " | ".join(order) + " |")
        print("| --- | ---: | --- | " + " | ".join("---:" for _ in order) + " |")
        for k in keys:
            cells = []
            for c in order:
                v = [runs[(s, c)][p][k] for p in sorted(runs[(s, c)]) if k in runs[(s, c)][p]]
                cells.append(" / ".join(f"{x:.1f}" for x in v) + (f" ({sum(v) / len(v):.1f})" if len(v) > 1 else ""))
            print(f"| {k[0]} | {k[1]} | {k[2].replace('NOC_UNICAST_', '').lower()} | " + " | ".join(cells) + " |")
        for c in order:
            for p in sorted(runs[(s, c)]):
                print(f"   slots {p}_{c}_{s}: {logs.get((s, c, p), '')}")


if __name__ == "__main__":
    main()
