#!/usr/bin/env python3
"""DRISC offload (15226240 = 13129088 + bit 21, dq_*) against its same-tree slow-dispatch baseline (dqb_13129088_*).
With --fd, the fast-dispatch rerun (dqf_* vs dqfb_*). With three prefixes (e.g. `dqg dqgb dqf`), offload <N> against
baseline <NB>, with offload <REF> (an earlier offload run) as the bracketed column. Every row is listed: baseline ->
offload (delta) cycles/packet and the bracketed reference: by default the fast-dispatch baseline from §37 (fpb_*) as a
tree check, and for W32 the old blind push (fpb_15226240w32_*, §37 tree). Latency: median net round trip."""
import csv, os, re, statistics, sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if len(args) == 3:
    N, NB, REF = args
elif "--fd" in sys.argv:
    N, NB, REF = "dqf", "dqfb", None  # the fast-dispatch rerun
else:
    N, NB, REF = "dq", "dqb", None
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load_bw(label):
    f = os.path.join(R, label + ".csv")
    if not os.path.exists(f):
        return None
    out = {}
    for r in csv.DictReader(open(f)):
        k = (r["test_name"], r["ntype"], r.get("packet_size", r.get("size", "")))
        n = sum(1 for kk in out if kk[:3] == k)
        out[k + (n,)] = 1.35e9 / float(r["avg_packets_per_s"])  # n = 1 is the trimmed replay row
    return out


def load_lat(label):
    f = os.path.join(R, label + "_latency.csv")
    if not os.path.exists(f):
        return None
    groups = {}
    for r in csv.DictReader(open(f)):
        groups.setdefault(re.sub(r"_sample\d+$", "", r["test_name"]), []).append(float(r["net_avg_ns"]))
    return {k: statistics.median(v) for k, v in groups.items()}


def row_name(k):
    plain = "Mix" in k[0] or "Pure" in k[0]
    return (
        k[0]
        + ("" if plain else " " + k[1][4:])
        + (" " + k[2] if k[2] and not plain else "")
        + (" (replay)" if k[3] else "")
    )


def table(title, new, base, ref, ref_tag):
    a, b, c = load_bw(new), load_bw(base), load_bw(ref)
    print(f"\n{title}: {base} -> {new}   [{ref_tag}: {ref}]")
    if a is None or b is None:
        print("  missing")
        return
    for k in b:
        if k not in a:
            print(f"  {row_name(k):58s} {b[k]:7.1f} ->   (missing)")
            continue
        extra = f"   [{c[k]:7.1f}]" if c and k in c else ""
        print(f"  {row_name(k):58s} {b[k]:7.1f} -> {a[k]:7.1f} ({a[k] - b[k]:+6.1f}){extra}")


for s in ("4352", "mix", "4352_trim", "mix_trim"):
    if REF:
        table(s, f"{N}_15226240_{s}", f"{NB}_13129088_{s}", f"{REF}_15226240_{s}", "earlier offload")
    else:
        table(s, f"{N}_15226240_{s}", f"{NB}_13129088_{s}", f"fpb_13129088_{s}", "FD baseline")
for s in ("4352", "mix"):
    if REF:
        table(f"W32 {s}", f"{N}_15226240w32_{s}", f"{NB}_13129088_{s}", f"{REF}_15226240w32_{s}", "earlier offload W32")
    else:
        table(f"W32 {s}", f"{N}_15226240w32_{s}", f"{NB}_13129088_{s}", f"fpb_15226240w32_{s}", "old blind push W32")
if REF:
    table("small", f"{N}_15226240_small", f"{NB}_13129088_small", f"{REF}_15226240_small", "earlier offload")
else:
    table("small", f"{N}_15226240_small", f"{NB}_13129088_small", "fp_13129088_small", "FD baseline")
ref_lat = f"{REF}_15226240_lat" if REF else "fp_13129088_lat"
a, b, c = load_lat(f"{N}_15226240_lat"), load_lat(f"{NB}_13129088_lat"), load_lat(ref_lat)
print(f"\nlatency (median net round trip, ns): {NB}_13129088 -> {N}_15226240   [{ref_lat}]")
if a and b:
    for k in b:
        if k in a:
            extra = f"   [{c[k]:7.1f}]" if c and k in c else ""
            print(f"  {k:58s} {b[k]:7.1f} -> {a[k]:7.1f} ({a[k] - b[k]:+6.1f}){extra}")
