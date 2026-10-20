#!/usr/bin/env python3
"""Summary table for bits 26/27/28: cycles/packet for grouped rows, each build against its own same-tree baseline.
Default: the §36 rework (fn_* vs fnb_*) next to §35 (fm_* vs fmb_*). With --fp: the §37 version (fp_* vs fpb_*) next
to §36 (fn_*), bits 26 / 28 / 26+28 only."""
import csv, os, statistics, sys

FP = "--fp" in sys.argv
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(label):
    f = os.path.join(R, label + ".csv")
    if not os.path.exists(f):
        return None
    out = {}
    for r in csv.DictReader(open(f)):
        k = (r["test_name"], r["ntype"], r.get("packet_size", r.get("size", "")))
        n = sum(1 for kk in out if kk[:3] == k)
        out[k + (n,)] = 1.35e9 / float(r["avg_packets_per_s"])
    return out


def pick(d, names, rep=0, ntype=None):
    v = [x for k, x in d.items() if k[0] in names and k[3] == rep and (ntype is None or k[1] == ntype)]
    return statistics.mean(v) if v else None


SMALL = lambda p: [f"{p}_PureWrite16", f"{p}_PureWrite256", f"{p}_PureWrite1024"]
W = "NOC_UNICAST_WRITE"
GROUPS = [  # (label, suite, names, replay, ntype)
    ("E/W small writes", "mix", SMALL("EW"), 0, None),
    ("N/S small writes", "mix", SMALL("NS"), 0, None),
    ("fwd small writes", "mix", SMALL("FWD"), 0, None),
    ("E/W MixSize2/4", "mix", ["EW_MixSize2", "EW_MixSize4"], 0, None),
    ("E/W write 4352 (mix)", "mix", ["EW_PureWrite4352"], 0, None),
    ("fwd write 4352 (mix)", "mix", ["FWD_PureWrite4352"], 0, None),
    (
        "scatter/fused/atomic, all dirs",
        "mix",
        [f"{p}_{t}" for p in ("EW", "NS", "FWD") for t in ("PureScatter4352", "PureFused4352", "PureAtomic")],
        0,
        None,
    ),
    ("1-hop E/W write 4352", "4352", ["Mesh1HopEW"], 0, W),
    ("1-hop N/S write 4352", "4352", ["Mesh1HopNS"], 0, W),
    ("3-hop E/W write 4352", "4352", ["Mesh3HopEW"], 0, W),
    ("multicast write 4352", "4352", ["MeshMulticast"], 0, W),
    ("E/W small, trimmed first pass", "mix_trim", SMALL("EW"), 0, None),
    ("E/W small, trimmed replay", "mix_trim", SMALL("EW"), 1, None),
    ("fwd small, trimmed first pass", "mix_trim", SMALL("FWD"), 0, None),
    ("fwd small, trimmed replay", "mix_trim", SMALL("FWD"), 1, None),
    ("3-hop 4352, trimmed first pass", "4352_trim", ["Mesh3HopEW"], 0, W),
    ("3-hop 4352, trimmed replay", "4352_trim", ["Mesh3HopEW"], 1, W),
]
CFG = [
    ("80237952", "+26"),
    ("147346816", "+27"),
    ("281564544", "+28"),
    ("348673408", "+26+28"),
    ("415782272", "+27+28"),
]
if FP:
    CFG = [c for c in CFG if c[0] in ("80237952", "281564544", "348673408")]
NEW, NEWB, OLD, OLDB, OLDTAG = ("fp", "fpb", "fn", "fnb", "§36") if FP else ("fn", "fnb", "fm", "fmb", "§35")


def cell(pre, bpre, f, suite, g):
    a, b = load(f"{pre}_{f}_{suite}"), load(f"{bpre}_13129088_{suite}")
    if a is None or b is None:
        return "", None
    x, y = pick(a, g[2], g[3], g[4]), pick(b, g[2], g[3], g[4])
    return (f"{x - y:+.1f}" if x is not None and y is not None else ""), y


print(f'{"cycles/packet":32s} {"base":>6s} ' + " ".join(f"{n:>13s}" for _, n in CFG))
print(f'{"":32s} {"":6s} ' + " ".join(f'{"new [" + OLDTAG + "]":>13s}' for _ in CFG))
for g in GROUPS:
    base = None
    cells = []
    for f, _ in CFG:
        new, b = cell(NEW, NEWB, f, g[1], g)
        old, _ = cell(OLD, OLDB, f, g[1], g)
        base = base or b
        cells.append(f"{new:>5s} [{old:>5s}]" if new or old else "")
    print(f"{g[0]:32s} {base or 0:6.1f} " + " ".join(f"{c:>13s}" for c in cells))
print("\nBlind push W32 (vs 15226240):")
for g in [
    ("E/W small writes", "mix", SMALL("EW"), 0, None),
    ("N/S small writes", "mix", SMALL("NS"), 0, None),
    ("fwd small writes", "mix", SMALL("FWD"), 0, None),
    ("1-hop E/W write 4352", "4352", ["Mesh1HopEW"], 0, W),
    ("1-hop N/S write 4352", "4352", ["Mesh1HopNS"], 0, W),
    ("E/W write 4352 (mix)", "mix", ["EW_PureWrite4352"], 0, None),
    ("N/S write 4352 (mix)", "mix", ["NS_PureWrite4352"], 0, None),
    ("fwd write 4352 (mix)", "mix", ["FWD_PureWrite4352"], 0, None),
]:
    out = []
    for f, n in (("82335104", "+26"),) if FP else (("82335104", "+26"), ("149443968", "+27")):
        a, b = load(f"{NEW}_{f}w32_{g[1]}"), load(f"{NEWB}_15226240w32_{g[1]}")
        oa, ob = load(f"{OLD}_{f}w32_{g[1]}"), load(f"{OLDB}_15226240w32_{g[1]}")
        x, y = pick(a, g[2], g[3], g[4]), pick(b, g[2], g[3], g[4])
        ox, oy = pick(oa, g[2], g[3], g[4]), pick(ob, g[2], g[3], g[4])
        out.append(f"{n} {x - y:+.1f} [{ox - oy:+.1f}]")
    print(f"  {g[0]:30s} base {y:6.1f}  " + "   ".join(out))
