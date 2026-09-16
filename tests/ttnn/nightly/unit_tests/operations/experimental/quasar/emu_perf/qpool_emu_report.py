#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Compare qpool_emu_perf.py legs (ZeBu emulator, device-profiler cycles of the pool2d program) and put
them next to the craq-sim deltas from PR #55001.

  python qpool_emu_report.py results/before/results.json results/after/results.json [results/after_t1/results.json]

Per case: median pool2d kernel cycles per leg (first *-KERNEL zone start to last *-KERNEL zone end over
all cores/RISCs of the program), delta% (NEGATIVE = right leg faster) and speedup. The PR's craq-sim
table is threads=1 -> threads=4 in the NEW tree (halo+pool dispatch envelope), so 'after_t1 -> after'
is the like-for-like methodology and 'before -> after' the true commit delta (old tree = two split
readers on two DM cores + one NEO).
"""

import json
import math
import statistics
import sys

# PR #55001 craq-sim table (clocks, threads=1 -> threads=4, halo+pool envelope)
SIM = {
    "avg_k7x7_s1_large": (223493, 61203),
    "k9x9_s2_3chunks": (577209, 176789),
    "k8x8_s2_large": (449398, 144400),
    "avg_k3x3_s1": (410842, 134565),
    "k7x7_s2_large": (357467, 121375),
    "k3x3_s1": (164407, 71869),
    "wide_c280_3blocks": (183520, 89078),
    "block_2x2": (49015, 29923),
    "k5x5_s2": (63934, 46695),
    "batch2": (51000, 43200),
    "tall_32x4": (51000, 43200),
    "wide_4x32": (51000, 43200),
    "width_1x2_c128": (51000, 43200),
}
SIM_ALIAS = {  # emulator case -> sim case it stands in for
    "block_1x2_c128": "block_2x2",
    "k5x5_s2_2c32": "k5x5_s2",
    "batch2_4x8_2c32": "batch2",
    "tall_16x4_2c32": "tall_32x4",
    "wide_4x16_2c32": "wide_4x32",
}


def load(path):
    with open(path) as f:
        d = json.load(f)
    out = {}
    for name, r in d["cases"].items():
        if "iters" not in r or not r["iters"]:
            out[name] = dict(error=r.get("error", "no iters"), note=r.get("note"))
            continue
        pools = [it["programs"][-1]["kernel_cycles"] for it in r["iters"] if it["programs"] and it["programs"][-1]["kernel_cycles"]]
        # per-RISC kernel spans of the last iteration's pool program, grouped by DM lane / NEO
        riscs = r["iters"][-1]["programs"][-1]["riscs"] if r["iters"][-1]["programs"] else {}
        dm = sorted((k, v.get("cycles")) for k, v in riscs.items() if "_DM" in k and v.get("cycles"))
        trisc = sorted((k, v.get("cycles")) for k, v in riscs.items() if "TRISC" in k and v.get("cycles", 0) > 1000)
        out[name] = dict(
            pool=statistics.median(pools) if pools else None,
            pool_all=sorted(pools),
            nprog=sorted({len(it["programs"]) for it in r["iters"]}),
            verdict=r.get("verdict"),
            cores=r.get("core_desc"),
            note=r.get("note"),
            dm=dm,
            trisc=trisc,
        )
    return d["meta"], out


def fmt_pair(a, b):
    if not a or not b:
        return f"{'-':>10} {'-':>10} {'-':>8} {'-':>7}"
    return f"{a:>10.0f} {b:>10.0f} {(b - a) / a * 100:>+7.1f}% {a / b:>6.2f}x"


def geomean(ratios):
    ratios = [r for r in ratios if r]
    return math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if ratios else float("nan")


def main(argv):
    legs = [load(p) for p in argv[1:]]
    names = [m["leg"] for m, _ in legs]
    for m, _ in legs:
        print(f"leg {m['leg']:<10} tree={m['tree']} git='{m['git']}' grid={m.get('grid')} iters={m['iters']}")
    cases = list(legs[0][1].keys())
    for _, c in legs[1:]:
        cases += [k for k in c if k not in cases]

    # legs[0] = baseline (before), legs[1] = after; extra legs: a T=1 leg is compared AS a baseline of
    # after (t1 -> after), any other diagnostic leg is compared AGAINST after (after -> leg).
    pairs = [(0, 1)] + [((k, 1) if "t1" in names[k] else (1, k)) for k in range(2, len(legs))]
    print("\n== pool2d program kernel cycles on the ZeBu emulator (median of iters; delta% NEGATIVE = right leg faster) ==")
    hdr = f"{'case':<22}"
    for a, b in pairs:
        hdr += f" | {names[a]:>10} {names[b]:>10} {'delta%':>8} {'speedup':>7}"
    hdr += f" | {'sim T1->T4':>10}  cores (emu)"
    print(hdr)
    ratios = {p: [] for p in pairs}
    sim_ratios = {p: [] for p in pairs}
    for c in cases:
        row = f"{c:<22}"
        sim = SIM.get(SIM_ALIAS.get(c, c))
        for a, b in pairs:
            va, vb = legs[a][1].get(c, {}).get("pool"), legs[b][1].get(c, {}).get("pool")
            row += " | " + fmt_pair(va, vb)
            if va and vb:
                ratios[(a, b)].append(va / vb)
                if sim and c in SIM:
                    sim_ratios[(a, b)].append((va / vb, sim[0] / sim[1]))
        sim_s = f"{sim[0] / sim[1]:>9.2f}x" if sim else f"{'-':>10}"
        row += f" | {sim_s}  {legs[-1][1].get(c, {}).get('cores', '')}"
        print(row)
    for (a, b), rs in ratios.items():
        print(f"  geomean speedup {names[a]} -> {names[b]}: {geomean(rs):.2f}x over {len(rs)} cases")
        pr = sim_ratios[(a, b)]
        if pr:
            print(
                f"    on the {len(pr)} PR-table cases: emulator geomean {geomean([e for e, _ in pr]):.2f}x "
                f"vs craq-sim geomean {geomean([s for _, s in pr]):.2f}x"
            )

    # Fixed-vs-per-stick decomposition from same-kernel pairs at 32 and 64 sticks/core (3x3 s2 C=64, 2 cores):
    # cycles = fixed + per_stick * sticks  ->  fixed = 2*c32 - c64, per_stick = (c64 - c32) / 32
    print("\n== per-program fixed cost vs per-stick cost (3x3 s2 C=64, 2 cores; from the 32- and 64-stick/core twins) ==")
    print(f"{'pair':<34}" + "".join(f" | {m['leg']:>9} fixed {'per-stick':>9}" for m, _ in legs))
    for c32, c64 in (("batch2_4x8_2c32", "batch2"), ("tall_16x4_2c32", "tall_32x4"), ("wide_4x16_2c32", "wide_4x32")):
        row = f"{c32 + ' / ' + c64:<34}"
        for m, d in legs:
            a, b = d.get(c32, {}).get("pool"), d.get(c64, {}).get("pool")
            row += f" | {2 * a - b:>15.0f} {(b - a) / 32:>9.0f}" if a and b else f" | {'-':>15} {'-':>9}"
        print(row)

    print("\n== iteration spread (all measured iterations, cycles) ==")
    for c in cases:
        print(f"{c:<22} " + " | ".join(f"{m['leg']}: {d.get(c, {}).get('pool_all')}" for m, d in legs))

    print("\n== lane balance, last iteration (per-RISC kernel span, cycles; DM = reader lanes, TRISC = NEO compute) ==")
    for c in cases:
        for m, d in legs:
            r = d.get(c, {})
            if not r.get("dm"):
                continue
            dm = " ".join(f"{k.split(':')[1].replace('QUASAR_', '')}={v}" for k, v in r["dm"] if k.startswith("(0,1)"))
            neo = {}
            for k, v in r["trisc"]:
                if k.startswith("(0,1)"):
                    neo.setdefault(k.split("_TRISC")[0].split("QUASAR_")[1], []).append(v)
            neo_s = " ".join(f"{n}={max(v)}" for n, v in sorted(neo.items()))
            print(f"{c:<22} {m['leg']:<9} core(0,1): {dm} | {neo_s}")

    print("\n== verdicts / notes ==")
    for c in cases:
        parts = [f"{m['leg']}: {d.get(c, {}).get('verdict') or d.get(c, {}).get('error', '-')}" for m, d in legs]
        print(f"{c:<22} {' | '.join(parts)}")
        note = legs[-1][1].get(c, {}).get("note")
        if note:
            print(f"{'':<22} note: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
