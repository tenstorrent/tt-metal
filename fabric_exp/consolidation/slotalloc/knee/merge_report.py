#!/usr/bin/env python3
"""merge_report.py [--results DIR ...] [--prefix kn] [--sets M24,M42,MG,L1]: the completion-group merge sweep
(SLOT_ALLOCATOR.md sections 12.10 and 12.11; sets M24 / M42 / MG / L1 of knee_arms.sh).

Each timing arm is w<W>f<F>r<R>g<G>: worker W, forwarding F, receiver R slots, TT_FABRIC_EXP_COMPLETION_GROUP=G. Each
merge test k<N> runs N far sender channels into one receiver (k1 the worker alone, k1f one forwarding channel alone).
Per test, packet size and flow (the sender's chip): cycles/packet from the run log's per-sender lines (the sender
kernel's own cycle count), mean over the two passes; the link total is 1 / sum(1 / flow). Tables: the link total per (W, F, R) and k at G1 / G2 / G4 with G1 and G2 against G4; the
per-flow rates; the pass spread; and the group each rule picks (today: from R; A: 4 when every feeding channel and R
are >= 16, 1 when any is < 8, else 2; B: min(4, pow2_floor(min(R, smallest feeding) / 4))) with that rule's link total
against today's.

P150 labels <prefix><a|b>_<set>_<arm> (knee_p150_runs.sh); Galaxy labels knm_<a|b>_<arm>_row_g<G> (dyn_glx.sbatch).
"""
import argparse
import collections
import csv
import os
import re
import statistics as st
import subprocess

FLOWS = {"M24": {5: "worker", 4: "straight", 1: "turnN"},
         "M42": {2: "worker", 0: "turnN", 4: "turnS"},
         "MG": {13: "worker", 12: "straight", 9: "turnN", 17: "turnS"},
         "L1": {1: "worker", 0: "pass-through"}}
SETS = ("M24", "M42", "MG", "L1")
HERE = os.path.dirname(os.path.abspath(__file__))
ARM = re.compile(r"^w(\d+)f(\d+)r(\d+)g([124])$")


def arms_of(s):
    """The set's timing arms (w<W>f<F>r<R>g<G>) from knee_arms.sh."""
    out = subprocess.run(["bash", "-c", f"source {HERE}/knee_arms.sh; knee_arms"], capture_output=True, text=True).stdout
    return [f[1] for f in (line.split() for line in out.splitlines()) if len(f) > 1 and f[0] == s and ARM.match(f[1])]


def pow2_floor(n):
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def rule_today(w, f, r):
    return 1 if r < 8 else (2 if r < 16 else 4)


def rule_a(w, f, r):
    feeding = (w, f, r)
    return 4 if min(feeding) >= 16 else (1 if min(feeding) < 8 else 2)


def rule_b(w, f, r):
    return min(4, pow2_floor(max(1, min(r, w, f) // 4)))


def flows_of(path):
    """{(test, size): {chip: cycles/packet}} from a run's log: the bandwidth profiler's per-sender lines ("Sender core
    ... on device D ...: P packets in C cycles") under each "Running Test: <name>" line. (The run's CSV row is the
    busiest link with only its own chip's sender cycles, so it does not give the forwarded flows.)"""
    out = collections.defaultdict(dict)
    clean = re.compile(r"\x1b\[[0-9;]*m")
    key = None
    for line in open(path, errors="replace"):
        line = clean.sub("", line)
        m = re.search(r"Running Test: (\S+)_size_(\d+)_", line)
        if m:
            key = (m.group(1).split("_")[-1], int(m.group(2)))
            continue
        m = re.search(r"(?:Sender core|another sender core) \S+ \(VC\d\) (?:on device (\d+)|with).*?(\d+) packets in (\d+) cycles", line)
        if m and key and m.group(1) is not None:
            out[key][int(m.group(1))] = int(m.group(3)) / int(m.group(2))
    return out


def find(dirs, name):
    for d in dirs:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="+", default=["/home/aagarwal/tt-metal/fabric_exp/results"])
    ap.add_argument("--prefix", default="kn")
    ap.add_argument("--sets", default=",".join(SETS))
    a = ap.parse_args()
    for s in a.sets.split(","):
        data, spread = {}, 0.0  # (w, f, r, g) -> {(test, size): {chip: cpp}}
        for arm in arms_of(s):
            w, f, r, g = map(int, ARM.match(arm).groups())
            names = ([f"knm_{p}_{arm}_row_g{g}.log" for p in "ab"] if s == "MG"
                     else [f"{a.prefix}{p}_{s}_{arm}.log" for p in "ab"])
            passes = [flows_of(p) for p in (find(a.results, n) for n in names) if p]
            if not passes:
                continue
            keys = set.intersection(*(set(d) for d in passes))
            cell = {}
            for k in keys:
                chips = set.intersection(*(set(d[k]) for d in passes))
                cell[k] = {c: st.mean(d[k][c] for d in passes) for c in chips}
                if len(passes) == 2:
                    spread = max(spread, max(abs(passes[0][k][c] - passes[1][k][c]) for c in chips))
            cell["passes"] = len(passes)
            data[(w, f, r, g)] = cell
        if not data:
            print(f"## {s}: no results\n")
            continue
        tests = sorted({k for c in data.values() for k in c if k != "passes"}, key=lambda t: (t[1], t[0]))
        rows = sorted({k[:3] for k in data})
        link = lambda row, g, t: (lambda c: 1.0 / sum(1.0 / x for x in c.values()) if c else None)(  # noqa: E731
            data.get(row + (g,), {}).get(t))
        ks = sorted({t[0] for t in tests}, key=lambda n: (len(n), n))
        print(f"## {s}: summary, link total cycles/packet G1 / G2 / G4 (mean of two passes)\n")
        print(f"passes: {sorted({c['passes'] for c in data.values()})}; largest pass spread {spread:.1f} cycles\n")
        for size in sorted({t[1] for t in tests}):
            print(f"| {size} B: W,F,R | " + " | ".join(ks) + " |")
            print("| --- | " + " | ".join("---" for _ in ks) + " |")
            for row in rows:
                cells = []
                for k in ks:
                    v = [link(row, g, (k, size)) for g in (1, 2, 4)]
                    cells.append(" / ".join(f"{x:.1f}" if x else "-" for x in v))
                print(f"| {','.join(map(str, row))} | " + " | ".join(cells) + " |")
            print()
        print(f"## {s}: link total, cycles/packet (G1 / G2 / G4; G1 and G2 against G4)\n")
        print("| W,F,R | test | size | G1 | G2 | G4 | G1 vs G4 | G2 vs G4 |")
        print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        totals = {}
        for row in rows:
            for t in tests:
                v = {}
                for g in (1, 2, 4):
                    c = data.get(row + (g,), {}).get(t)
                    if c:
                        v[g] = 1.0 / sum(1.0 / x for x in c.values())
                totals[(row, t)] = v
                if not v:
                    continue
                cell = lambda g: f"{v[g]:.1f}" if g in v else "-"  # noqa: E731
                rel = lambda g: f"{100 * (v[g] / v[4] - 1):+.1f}%" if g in v and 4 in v else "-"  # noqa: E731
                print(f"| {','.join(map(str, row))} | {t[0]} | {t[1]} | {cell(1)} | {cell(2)} | {cell(4)} | "
                      f"{rel(1)} | {rel(2)} |")
        print(f"\n## {s}: per flow, cycles/packet (G1 / G2 / G4)\n")
        names = FLOWS.get(s, {})
        print("| W,F,R | test | size | flow | G1 | G2 | G4 |")
        print("| --- | --- | ---: | --- | ---: | ---: | ---: |")
        for row in rows:
            for t in tests:
                chips = sorted({c for g in (1, 2, 4) for c in data.get(row + (g,), {}).get(t, {})})
                if len(chips) < 2:
                    continue
                for c in chips:
                    vals = [data.get(row + (g,), {}).get(t, {}).get(c) for g in (1, 2, 4)]
                    print(f"| {','.join(map(str, row))} | {t[0]} | {t[1]} | {names.get(c, c)} | "
                          + " | ".join(f"{x:.1f}" if x else "-" for x in vals) + " |")
        print(f"\n## {s}: rules (link total; against today's group)\n")
        print("| W,F,R | test | size | today G | A G | B G | today | A | B | A vs today | B vs today |")
        print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in rows:
            gt, ga, gb = rule_today(*row), rule_a(*row), rule_b(*row)
            for t in tests:
                v = totals.get((row, t), {})
                if gt not in v:
                    continue
                f = lambda g: f"{v[g]:.1f}" if g in v else "-"  # noqa: E731
                rel = lambda g: f"{100 * (v[g] / v[gt] - 1):+.1f}%" if g in v else "-"  # noqa: E731
                print(f"| {','.join(map(str, row))} | {t[0]} | {t[1]} | {gt} | {ga} | {gb} | {f(gt)} | {f(ga)} | "
                      f"{f(gb)} | {rel(ga)} | {rel(gb)} |")
        print()


if __name__ == "__main__":
    main()
