#!/usr/bin/env python3
"""knee_report.py [--results DIR ...] [--prefix kn] [--sets A,B,...] [--tol 1.0]: the slot-need sweep's knees.

Reads the runs knee_p150_runs.sh / dyn_glx.sbatch wrote (P150 labels <prefix><a|b>_<set>_<arm>; Galaxy labels
<kind>_<a|b>_<arm>_row) and the arm table (knee_arms.sh). Arms that vary the same depth of the same suite form a sweep:
w* worker (W), r* receiver (R), f* forwarding (F), v<F>r8 / x44v<F>r8 VC1 forwarding (F1), v6r<R> / x44v6r<R> VC1
receiver (R1); check arms (f12w29, x44chk) and the squeeze (sq*) and validation (val*) arms are reported apart.
Per test row and packet size: cycles/packet (1.35 GHz), mean over the two passes and over the row's senders (mixed
rows: link total, 1 / sum(1 / flow)). The knee = the smallest swept depth within TOL % of the sweep's best, shown as
the interval the sweep bounds it to ("17-20"); ">16" when the largest depth swept still gains past TOL over the one
before it (no upper bound). Per sweep and packet size: the largest knee over its rows
(the channel serves them all) and the row that sets it."""
import argparse
import collections
import csv
import glob
import os
import re
import statistics as st
import subprocess

CLK = 1.35e9
MIX = re.compile(r"Mix")
HERE = os.path.dirname(os.path.abspath(__file__))


def arms():
    out = subprocess.run(["bash", "-c", f"source {HERE}/knee_arms.sh; knee_arms"], capture_output=True, text=True).stdout
    return [line.split()[:5] for line in out.splitlines() if line.strip()]


def sweep_of(arm, row):
    v = row.split(",") if row != "-" else []
    for pat, var, idx in ((r"^x44v(\d+)r8$", "VC1 forwarding", 3), (r"^x44v6r(\d+)$", "VC1 receiver", 4),
                          (r"^v(\d+)r8$", "VC1 forwarding", 3), (r"^v6r(\d+)$", "VC1 receiver", 4),
                          (r"^v10r(\d+)$", "VC1 receiver", 4), (r"^x44v10r(\d+)$", "VC1 receiver", 4),
                          (r"^w(\d+)(_[0-9a-z]+)?$", "worker", 0), (r"^r(\d+)(_[0-9a-z]+)?$", "receiver", 2),
                          (r"^f(\d+)(_[0-9a-z]+)?$", "forwarding", 1)):
        if re.match(pat, arm):
            return var, int(v[idx])
    return None, None


def rows_of(path):
    acc = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        pps = float(r["packets_per_second"])
        if pps <= 0:
            continue
        name = re.sub(r"_size_\d+.*$", "", r["test_name"])
        acc[(name, int(r["packet_size"]))].append(CLK / pps)
    out = {}
    for (name, size), v in acc.items():
        out[(name, size)] = 1.0 / sum(1.0 / c for c in v) if MIX.search(name) else st.mean(v)
    return out


def find(dirs, names):
    for d in dirs:
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", default=["/home/aagarwal/tt-metal/fabric_exp/results"])
    ap.add_argument("--prefix", default="kn")
    ap.add_argument("--sets", default="")
    ap.add_argument("--tol", type=float, default=1.0)
    a = ap.parse_args()
    want = set(a.sets.split(",")) if a.sets else None
    galaxy_kind = {"g2w_4352": "kn2w", "g2f_4352": "kn2f", "gvc1_4352": "vc1kn", "gvc1x44_4352": "vc1knx44",
                   "gvc1x44san_4352": "vc1x44san",
                   "gvc1_3072": "vc1kn3", "gvc1x44_3072": "vc1knx443", "gvc1x44san_3072": "vc1x44san3"}
    data = {}  # (set, arm) -> {(row, size): cpp}
    for s, arm, kind, suite, row in arms():
        if want and s not in want:
            continue
        per_pass = []
        for p in "ab":
            if kind.startswith("g"):
                names = [f"{galaxy_kind.get(suite, '?')}_{p}_{arm}_row_tests.csv", f"{galaxy_kind.get(suite, '?')}_{p}_{arm}_def_tests.csv"]
            else:
                names = [f"{a.prefix}{p}_{s}_{arm}_tests.csv", f"{a.prefix}{p}_{s}_{arm}_13129088_vc1_tests.csv"]
            f = find(a.results, names)
            if f:
                per_pass.append(rows_of(f))
        if per_pass:
            keys = set.intersection(*(set(d) for d in per_pass))
            data[(s, arm)] = {k: st.mean(d[k] for d in per_pass) for k in keys}
            data[(s, arm, "passes")] = len(per_pass)
    sweeps = collections.defaultdict(list)  # (set, suite, var) -> [(depth, arm, row)]
    others = []
    for s, arm, kind, suite, row in arms():
        if (s, arm) not in data:
            continue
        var, depth = sweep_of(arm, row)
        if var:
            sweeps[(s, suite, var)].append((depth, arm, row))
            if (s not in ("C2", "G3") and re.match(r"^(x44)?v6r8$", arm)) or (s in ("C2", "G3") and re.match(r"^(x44)?v10r8$", arm)):  # shared point
                sweeps[(s, suite, "VC1 receiver")].append((int(row.split(",")[4]), arm, row))
        else:
            others.append((s, arm, suite, row))
    for (s, suite, var), pts in sorted(sweeps.items()):
        pts = sorted(set(pts))
        if len(pts) < 2:
            continue
        print(f"\n## {s} {suite}: {var} ({', '.join(f'{d}' for d, _, _ in pts)}; rows {pts[0][2]} .. {pts[-1][2]};"
              f" passes {min(data[(s, arm, 'passes')] for _, arm, _ in pts)})")
        keys = set.intersection(*(set(data[(s, arm)]) for _, arm, _ in pts))
        sizes = sorted({k[1] for k in keys})
        print("| row | size | " + " | ".join(str(d) for d, _, _ in pts) + " | knee |")
        print("| --- | ---: | " + " | ".join("---:" for _ in pts) + " | ---: |")
        knee_by_size = collections.defaultdict(list)
        for k in sorted(keys):
            vals = [data[(s, arm)][k] for _, arm, _ in pts]
            best = min(vals)
            knee_i = next(i for i, v in enumerate(vals) if v <= best * (1 + a.tol / 100))
            # The knee as the interval the sweep bounds it to: "17-20" = more than the previous depth, at most this
            # one; ">16" = the largest depth swept still gains past the threshold, so no upper bound.
            lo = pts[knee_i - 1][0] + 1 if knee_i > 0 else None
            if knee_i == len(pts) - 1 and knee_i > 0 and vals[-1] < vals[-2] * (1 - a.tol / 100):
                mark, knee = ">", f">{pts[knee_i - 1][0]}"
            else:
                mark = ""
                knee = f"{pts[knee_i][0]}" if lo is None or lo == pts[knee_i][0] else f"{lo}-{pts[knee_i][0]}"
            knee_by_size[k[1]].append((pts[knee_i][0], mark, k[0], knee))
            print(f"| {k[0]} | {k[1]} | " + " | ".join(f"{v:.1f}" for v in vals) + f" | {knee} |")
        print("\nKnee per packet size (largest over the rows; the row that sets it):")
        for size in sizes:
            kn = max(knee_by_size[size], key=lambda t: (t[0], t[1] == ">"))
            print(f"- {size} B: {kn[3]} ({kn[2]}; rows' knees "
                  + ", ".join(t[3] for t in sorted(knee_by_size[size])) + ")")
    if others:
        print("\n## Other arms (checks, squeeze, validation): cycles/packet")
        for s, arm, suite, row in others:
            d = data[(s, arm)]
            print(f"- {s} {arm} ({suite}, {row}): " + "; ".join(f"{k[0]} {k[1]} B {v:.1f}" for k, v in sorted(d.items())))


if __name__ == "__main__":
    main()
