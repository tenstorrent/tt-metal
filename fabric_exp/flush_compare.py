#!/usr/bin/env python3
"""Grouped transaction-ID flush: cycles/packet per test, new run (g_*) against its baseline. Rows that move by
>= 0.5 cycles are listed; the rest are summarized as a count."""
import csv, os, sys

R = "/home/aagarwal/tt-metal/fabric_exp/results"
PAIRS = [
    ("g_13129088_mix", "x_13129088_mix"),
    ("g_13129088_mix_trim", "x_13129088_mix_trim"),
    ("g_4740480_mix", "m_4740480_mix"),
    ("g_4740480_mix_trim", "m_4740480_mix_trim"),
    ("g_15226240w32_mix", "x_15226240w32_mix"),
    ("g_15226240w32_mix_trim", "x_15226240w32_mix_trim"),
    ("g_546176_mix", "m_546176_mix"),
    ("g_13129088_4352", "b_13129088_4352"),
    ("g_13129088_4352_trim", "b_13129088_4352_trim"),
]


def load(label):
    f = os.path.join(R, label + ".csv")
    out = {}
    if not os.path.exists(f):
        return None
    for r in csv.DictReader(open(f)):
        k = (r["test_name"], r["ntype"])
        n = sum(1 for kk in out if kk[:2] == k)
        out[k + (n,)] = 1.35e9 / float(r["avg_packets_per_s"])  # n = 1 is the trimmed replay row
    return out


for new, base in PAIRS:
    a, b = load(new), load(base)
    if a is None or b is None:
        print(f'\n{new}: missing ({"new" if a is None else "baseline"})')
        continue
    print(f"\n{new} vs {base}:")
    same = 0
    for k in a:
        if k not in b:
            continue
        d = a[k] - b[k]
        name = k[0] + ("" if "Mix" in k[0] or "Pure" in k[0] else " " + k[1]) + (" (replay)" if k[2] else "")
        if abs(d) >= 0.5 or "Atomic" in k[0] or "MixCmd" in k[0]:
            print(f"  {name:45s} {b[k]:7.1f} -> {a[k]:7.1f} ({d:+.1f})")
        else:
            same += 1
    print(f"  {same} other rows within 0.5")
