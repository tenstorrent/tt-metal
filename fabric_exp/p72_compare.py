#!/usr/bin/env python3
"""§72: the current tree (the user's s71) on the main rows: cycles/packet and payload GB/s with no profiling, slack with
a +20 spin on E0 (TX) and on E1 (RX), and per-packet phase costs on each ERISC from single-phase level-2 runs.
Net phase cost = (ticks - 4.0 x occurrences) / packets. E0 = the TX ERISC that sent the most packets; E1 rx = the RX
ERISC with the most local deliveries, E1 fwd = the one with the most forwards (if >= 1000)."""
import csv, os, re

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CAL = 4.0
E0 = {1: "tx_send", 2: "tx_txq_spin", 4: "tx_completion", 8: "tx_conn", 1024: "tx_visit"}
E1 = {16: "rx_decode_admit", 32: "rx_forward", 64: "rx_local", 128: "rx_completion", 2048: "rx_visit"}


def cyc(label):
    f = os.path.join(R, label + ".csv")
    return (
        {key(r["test_name"], r["ntype"]): 1.35e9 / float(r["avg_packets_per_s"]) for r in csv.DictReader(open(f))}
        if os.path.exists(f)
        else {}
    )


def key(test, ntype=""):
    # exp_2d_4352 names a write and a scatter test alike in the bandwidth CSV; the phase CSV spells the ntype out
    if "_ntype_" in test:
        base, nt = test.split("_size")[0], test.split("_ntype_")[1].split("_num_links")[0]
    else:
        base, nt = test, ntype.replace("NOC_UNICAST_", "").lower()
    return base + (" scatter" if "scatter" in nt else "") if base.startswith("Mesh") else base


def payload(t):
    if "Atomic" in t:
        return 0
    if t.startswith("Mesh"):
        return 4352
    m = re.search(r"(4352|16)$", t.split("_size")[0]) or re.search(r"size_(\d+)", t)
    return int(m.group(1)) if m else 4352


def phase(label, ph, erisc, col):
    f = os.path.join(R, label + "_phase.csv")
    if not os.path.exists(f):
        return {}
    out = {}
    rows = [r for r in csv.DictReader(open(f)) if r["erisc"] == erisc]
    for t in {r["test"] for r in rows}:
        r = max((r for r in rows if r["test"] == t), key=lambda r: int(r.get(col, 0) or 0))
        pk = int(r.get(col, 0) or 0)
        if pk < 1000:
            continue
        ticks, cnt = int(r.get(f"{ph}_ticks", 0) or 0), int(r.get(f"{ph}_count", 0) or 0)
        out[key(t)] = ((ticks - CAL * cnt) / pk, cnt / pk)
    return out


for n in ("p72", "4352"):
    base, tx, rx = (cyc(f"p72_{n}_{c}") for c in ("noprof", "txd20", "rxd20"))
    if not base:
        continue
    print(f"== {n}: cycles/packet, payload GB/s, slack (+20 spin; a fully bound ERISC moves ~+23 to +31)")
    print(f"   {'test':44s}{'cyc':>8s}{'GB/s':>8s}{'TX+20':>8s}{'RX+20':>8s}  binds")
    for t, c in base.items():
        p = payload(t)
        dt, dr = tx.get(t, float("nan")) - c, rx.get(t, float("nan")) - c
        binds = "E0" if dt > 2 * dr + 3 else "E1" if dr > 2 * dt + 3 else "both"
        gb = f"{p * 1.35 / c:8.1f}" if p else f"{1350 / c:6.2f}Mp"
        print(f"   {t[:44]:44s}{c:8.1f}{gb:>8s}{dt:+8.1f}{dr:+8.1f}  {binds}")
    for side, masks, roles in (
        ("E0", E0, {"tx": "tx_send_count"}),
        ("E1", E1, {"rx": "rx_local_count", "fwd": "rx_forward_count"}),
    ):
        for role, col in roles.items():
            d = {ph: phase(f"p72_{n}_m{m}", ph, "0" if side == "E0" else "1", col) for m, ph in masks.items()}
            tests = sorted(set().union(*[set(v) for v in d.values()]))
            if not tests:
                continue
            print(f"\n   {side} {role}: net cycles per packet (occurrences per packet)")
            print(f"   {'test':30s}" + "".join(f"{ph:>20s}" for ph in masks.values()))
            for t in tests:
                print(
                    f"   {t[:30]:30s}"
                    + "".join(
                        f"{d[ph][t][0]:12.1f} ({d[ph][t][1]:4.2f})" if t in d[ph] else f"{'-':>20s}"
                        for ph in masks.values()
                    )
                )
    print()
