#!/usr/bin/env python3
"""§68: §62 behavior (s62x) against §63 + guard (s63g) on streams/p68_profile.yaml: cycles/packet with no profiling,
E1 slack (RX spin +20 / +40), and per-packet E1 phase costs from single-phase level-2 runs. Net phase cost =
(ticks - 4.0 x occurrences) / packets, on the E1 with the most local deliveries (rx) and, for the forwarded rows, the E1 with the most
forwards (fwd)."""
import csv, os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
VS = ["s62x", "s63g"]
CAL = 4.0
MASKS = {16: "rx_decode_admit", 32: "rx_forward", 64: "rx_local", 128: "rx_completion", 2048: "rx_visit"}


def cyc(label):
    f = os.path.join(R, label + ".csv")
    return (
        {r["test_name"]: 1.35e9 / float(r["avg_packets_per_s"]) for r in csv.DictReader(open(f))}
        if os.path.exists(f)
        else {}
    )


def phase(label, ph):
    f = os.path.join(R, label + "_phase.csv")
    if not os.path.exists(f):
        return {}
    out = {}
    rows = [r for r in csv.DictReader(open(f)) if r["erisc"] == "1"]
    for t in sorted({r["test"] for r in rows}):
        rr = [r for r in rows if r["test"] == t]
        roles = {"rx": "rx_local_count", "fwd": "rx_forward_count"}
        for role, col in roles.items():
            r = max(rr, key=lambda r: int(r.get(col, 0) or 0))
            if int(r.get(col, 0) or 0) < 1000:
                continue
            pk = int(r["pkts"])
            ticks, cnt = int(r.get(f"{ph}_ticks", 0) or 0), int(r.get(f"{ph}_count", 0) or 0)
            if pk and cnt:
                out[(t.split("_size")[0], role)] = ((ticks - CAL * cnt) / pk, cnt / pk, pk)
    return out


base = {v: cyc(f"p68_{v}_noprof") for v in VS}
tests = list(base["s62x"])
print("cycles/packet (no profiling) and E1 slack: RX spin +20 / +40 (a fully E1-bound row moves ~+23 / +46)")
print(f"   {'test':42s}" + "".join(f"{v + ' ' + c:>13s}" for v in VS for c in ("none", "+20", "+40")))
for t in tests:
    print(
        f"   {t[:42]:42s}"
        + "".join(f"{cyc(f'p68_{v}_{c}').get(t, float('nan')):13.1f}" for v in VS for c in ("noprof", "rxd20", "rxd40"))
    )
print("\nE1 net phase cost per packet (occurrences per packet), single-phase level-2 runs")
for m, ph in MASKS.items():
    d = {v: phase(f"p68_{v}_m{m}", ph) for v in VS}
    keys = sorted(set(d["s62x"]) | set(d["s63g"]))
    if not keys:
        print(f"   {ph}: no data")
        continue
    print(f"   {ph}")
    for k in keys:
        a, b = d["s62x"].get(k), d["s63g"].get(k)
        fmt = lambda x: f"{x[0]:7.1f} ({x[1]:.2f})" if x else f"{'-':>14s}"
        diff = f"{b[0] - a[0]:+6.1f}" if a and b else ""
        print(f"      {k[0][:30]:30s} {k[1]:4s} s62x {fmt(a)}  s63g {fmt(b)}  {diff}")
