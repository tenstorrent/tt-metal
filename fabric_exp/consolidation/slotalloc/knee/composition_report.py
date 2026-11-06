#!/usr/bin/env python3
"""composition_report.py [--prefix kq] [--results DIR]: the composition test (SLOT_ALLOCATOR.md section 12.7).
Per arm (set Q, knee_arms.sh) and row: cycles/packet (mean of passes a, b; spread), the rate as a fraction of the
"full" arm (full cycles / arm cycles), and for each two-channel squeeze the measured fraction against min() and the
product of the single-channel fractions."""
import argparse, csv, os, re, statistics as st
CLK = 1.35e9


def load(p):
    out = {}
    for r in csv.DictReader(open(p)):
        pps = float(r["packets_per_second"])
        if pps > 0:
            out.setdefault((re.sub(r"_size_\d+.*$", "", r["test_name"]), int(r["packet_size"])), []).append(CLK / pps)
    return {k: st.mean(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="kq")
    ap.add_argument("--results", default="/home/aagarwal/tt-metal/fabric_exp/results")
    ap.add_argument("--set", default="Q|Q2")
    a = ap.parse_args()
    arms = {}
    for f in sorted(os.listdir(a.results)):
        m = re.match(rf"{a.prefix}([ab])_(?:{a.set})_(\w+?)_tests\.csv$", f)
        if m:
            arms.setdefault(m.group(2), {})[m.group(1)] = load(os.path.join(a.results, f))
    mean = {arm: {k: st.mean(d[k] for d in v.values()) for k in set.intersection(*(set(d) for d in v.values()))}
            for arm, v in arms.items() if len(v) == 2}
    spread = {arm: max(abs(v["a"][k] - v["b"][k]) for k in mean[arm]) for arm, v in arms.items() if len(v) == 2}
    full = mean["full"]
    order = ["full", "w10", "r8", "w10r8", "w14r4", "f8", "f8r8", "w10f8r8"] + sorted(set(mean) - {
        "full", "w10", "r8", "w10r8", "w14r4", "f8", "f8r8", "w10f8r8", "valq"})
    print("| arm | row | 256 B cycles (fraction of full) | 4352 B cycles (fraction) | pass spread |")
    print("| --- | --- | --- | --- | --- |")
    frac = {}
    for arm in order:
        if arm not in mean:
            continue
        for row in sorted({k[0] for k in mean[arm]}):
            cells = []
            for size in (256, 4352):
                c = mean[arm].get((row, size))
                fr = full[(row, size)] / c if c and (row, size) in full else None
                frac[(arm, row, size)] = fr
                cells.append(f"{c:.1f} ({fr:.3f})" if c and fr else "-")
            print(f"| {arm} | {row} | {cells[0]} | {cells[1]} | {spread[arm]:.1f} |")
    print("\nPairs: measured fraction against min(single fractions) and their product")
    print("| pair | row | size | single fractions | both squeezed | min | product |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    pairs = [("w10r8", "w10", "r8"), ("f8r8", "f8", "r8q3"), ("w10f8r8", "f8r8", "w10q3")]
    for both, x, y in pairs:
        for row in sorted({k[0] for k in mean.get(both, {})}):
            for size in (256, 4352):
                fb, fx, fy = frac.get((both, row, size)), frac.get((x, row, size)), frac.get((y, row, size))
                if None in (fb, fx, fy):
                    continue
                print(f"| {both} = {x} + {y} | {row} | {size} | {fx:.3f}, {fy:.3f} | {fb:.3f} | {min(fx, fy):.3f} |"
                      f" {fx * fy:.3f} |")
    print("\nUnequal against equal at the same total (worker + receiver 18 slots): w14r4 vs w10r8")
    for row in ("Sls1HopEW", "Sls1HopEWBidir"):
        for size in (256, 4352):
            e, u = frac.get(("w10r8", row, size)), frac.get(("w14r4", row, size))
            if e and u:
                print(f"- {row} {size} B: equal {e:.3f}, unequal {u:.3f} ({100 * (u / e - 1):+.1f}% rate)")


if __name__ == "__main__":
    main()
