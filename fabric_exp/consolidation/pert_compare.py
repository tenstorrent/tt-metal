#!/usr/bin/env python3
"""pert_compare.py: the timing-perturbation sweep (pert_runs.sh, RESULTS §98), cycles/packet against injected delay.

For each base (v3tk, v3v6, v3w, v3w4) and injection point (A: right before the completion check; B: right after a
packet's local write is issued), the rows the early-completion effect showed up on, at d = 0 (the base build) and
d = 8..32 cycles. The 0 -> 8 step includes the spin's own code (it reads the cycle counter at least twice), so judge
smoothness on d = 8..32: a robust design changes little and without steps there. "range" is max - min over 8..32,
"step" the largest change between neighbouring delays over 8..32.
"""
import csv, os

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
ROWS = [("4352", "Mesh1HopEW", "write"), ("4352", "Mesh1HopNS", "write"), ("mix", "EW_PureWrite4352", "write"),
        ("mix", "NS_PureWrite4352", "write"), ("4352", "Mesh3HopEW", "write"), ("mix", "EW_PureScatter4352", "scatter"),
        ("mix", "NS_PureScatter4352", "scatter"), ("1d", "Linear1HopEW", "scatter"), ("mix", "EW_MixSize4", None)]
DELAYS = [0, 8, 16, 24, 32]


def load(label, suite):
    f = os.path.join(R, f"{label}_13129088_{suite}.csv")
    if not os.path.exists(f):
        return {}
    out = {}
    for r in csv.DictReader(open(f)):
        k = (r["test_name"], r["ntype"].replace("NOC_UNICAST_", "").lower())
        out.setdefault(k, 1.35e9 / float(r["avg_packets_per_s"]))
    return out


def value(label, suite, test, ntype):
    d = load(label, suite)
    for (t, n), v in d.items():
        if t == test and (ntype is None or n.startswith(ntype)):
            return v
    return None


for base in ("v3tk", "v3v6", "v3w", "v3w4"):
    for p in ("A", "B"):
        labels = [f"p20_{base}_0"] + [f"p20_{base}_p{p}{d}" for d in DELAYS[1:]]
        print(f"\n{base} perturbation {p} ({'before the completion check' if p == 'A' else 'after the local write issue'})")
        print("   %-26s" % "row" + "".join("%8s" % f"d={d}" for d in DELAYS) + "%8s%8s" % ("range", "step"))
        for suite, test, ntype in ROWS:
            vals = [value(l, suite, test, ntype) for l in labels]
            if all(v is None for v in vals):
                continue
            cells = "".join("%8.1f" % v if v is not None else "%8s" % "-" for v in vals)
            tail = [v for v in vals[1:] if v is not None]
            rng = max(tail) - min(tail) if len(tail) > 1 else 0.0
            step = max((abs(b - a) for a, b in zip(tail, tail[1:])), default=0.0)
            name = test + ("" if ntype in (None, "write") else " " + ntype)
            print(f"   {name:26s}{cells}{rng:8.1f}{step:8.1f}")
