#!/usr/bin/env python3
"""Bit 22 single-register ACK-counter flush: cycles/packet per test for the stock-flush baseline, the grouped
transaction-ID flush (g_*/gf_*, RESULTS §33) and the ACK-counter flush (ak_*). Rows where the ACK-counter run moves
by >= 0.5 cycles from the baseline, or from the grouped run, are listed; the rest are summarized as a count."""
import csv, os

R = "/home/aagarwal/tt-metal/fabric_exp/results"
# (ACK-counter run, stock baseline, grouped run or None)
SETS = [
    ("ak_13129088_mix", "x_13129088_mix", "g_13129088_mix"),
    ("ak_13129088_mix_trim", "x_13129088_mix_trim", "g_13129088_mix_trim"),
    ("ak_4740480_mix", "m_4740480_mix", "g_4740480_mix"),
    ("ak_4740480_mix_trim", "m_4740480_mix_trim", "g_4740480_mix_trim"),
    ("ak_15226240w32_mix", "x_15226240w32_mix", "g_15226240w32_mix"),
    ("ak_15226240w32_mix_trim", "x_15226240w32_mix_trim", "g_15226240w32_mix_trim"),
    ("ak_13129088_4352", "b_13129088_4352", "g_13129088_4352"),
    ("ak_13129088_4352_trim", "b_13129088_4352_trim", "g_13129088_4352_trim"),
    ("ak_4740480_4352", "n_4740480_4352", "gf_4740480_4352"),
    ("ak_15226240w32_4352", "b_15226240w32_4352", None),
]


def load(label):
    f = os.path.join(R, label + ".csv") if label else None
    if not f or not os.path.exists(f):
        return None
    out = {}
    for r in csv.DictReader(open(f)):
        k = (r["test_name"], r["ntype"])
        n = sum(1 for kk in out if kk[:2] == k)
        out[k + (n,)] = 1.35e9 / float(r["avg_packets_per_s"])  # n = 1 is the trimmed replay row
    return out


for new, base, grp in SETS:
    a, b, g = load(new), load(base), load(grp)
    if a is None or b is None:
        print(f'\n{new}: missing ({"new" if a is None else "baseline"})')
        continue
    print(f'\n{new}   (baseline {base}{", grouped " + grp if g else ""})')
    print(f'  {"":45s} {"base":>7s} {"grouped":>8s} {"ack":>7s} {"vs base":>8s}')
    same = 0
    for k in a:
        if k not in b:
            continue
        gv = g.get(k) if g else None
        d = a[k] - b[k]
        moved = abs(d) >= 0.5 or (gv is not None and abs(a[k] - gv) >= 0.5)
        name = k[0] + ("" if "Mix" in k[0] or "Pure" in k[0] else " " + k[1]) + (" (replay)" if k[2] else "")
        if moved or "Atomic" in k[0] or "MixCmd" in k[0]:
            gs = f"{gv:8.1f}" if gv is not None else f'{"-":>8s}'
            print(f"  {name:45s} {b[k]:7.1f} {gs} {a[k]:7.1f} {d:+8.1f}")
        else:
            same += 1
    print(f"  {same} other rows within 0.5 of both")
