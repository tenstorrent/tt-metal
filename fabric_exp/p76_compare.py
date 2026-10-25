#!/usr/bin/env python3
"""§76: 1D on the current tree against the stock router (p76_runs.sh). Cycles/packet and payload GB/s for stock at the
firmware keepalive ("main"), stock at keepalive 1000, and the current tree; slack with a +20 spin on E0 (TX) and on E1
(RX); per-packet phase costs on each ERISC from single-phase level-2 runs. Net phase cost = (ticks - 4.0 x
occurrences) / packets. E0 = the TX ERISC that sent the most packets; E1 rx = the RX ERISC with the most local
deliveries, E1 fwd = the one with the most forwards (if >= 1000)."""
import csv, os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CAL = 4.0
PAYLOAD = 4352
E0 = {1: "tx_send", 2: "tx_txq_spin", 4: "tx_completion", 8: "tx_conn", 1024: "tx_visit"}
E1 = {16: "rx_decode_admit", 32: "rx_forward", 64: "rx_local", 128: "rx_completion", 2048: "rx_visit"}


def key(test, ntype=""):
    # bandwidth CSV: test name + ntype column; phase CSV: <name>_..._ntype_<ntype>_... in the test name
    if "_ntype_" in test:
        base, nt = test.split("_size")[0].split("_num_links")[0], test.split("_ntype_")[1]
    else:
        base, nt = test, ntype.lower()
    return base + (" scatter" if "scatter" in nt else "")


def cyc(label):
    f = os.path.join(R, label + ".csv")
    if not os.path.exists(f):
        return {}
    return {key(r["test_name"], r["ntype"]): 1.35e9 / float(r["avg_packets_per_s"]) for r in csv.DictReader(open(f))}


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


main, ka, cur = cyc("p76_stock_kafw"), cyc("p76_stock_ka1000"), cyc("p76_cur")
tx, rx = cyc("p76_txd20"), cyc("p76_rxd20")
gb = lambda c: PAYLOAD * 1.35 / c
print("== 1D, 4352 B: cycles/packet (payload GB/s); slack = change with a +20 spin (a fully bound ERISC moves ~+27 to +31)")
print(f"   {'test':30s}{'main':>14s}{'main ka1000':>14s}{'current':>14s}{'vs main':>9s}{'TX+20':>8s}{'RX+20':>8s}  binds")
for t, c in cur.items():
    m, k = main.get(t, float("nan")), ka.get(t, float("nan"))
    dt, dr = tx.get(t, float("nan")) - c, rx.get(t, float("nan")) - c
    binds = "E0" if dt > 2 * dr + 3 else "E1" if dr > 2 * dt + 3 else "both"
    print(
        f"   {t[:30]:30s}{m:7.1f} ({gb(m):4.1f}){k:7.1f} ({gb(k):4.1f}){c:7.1f} ({gb(c):4.1f}){(m / c - 1) * 100:+8.0f}%"
        f"{dt:+8.1f}{dr:+8.1f}  {binds}"
    )
for side, masks, roles in (
    ("E0", E0, {"tx": "tx_send_count"}),
    ("E1", E1, {"rx": "rx_local_count", "fwd": "rx_forward_count"}),
):
    for role, col in roles.items():
        d = {ph: phase(f"p76_m{m}", ph, "0" if side == "E0" else "1", col) for m, ph in masks.items()}
        tests = sorted(set().union(*[set(v) for v in d.values()]))
        if not tests:
            continue
        print(f"\n   {side} {role}: net cycles per packet (occurrences per packet)")
        print(f"   {'test':30s}" + "".join(f"{ph:>20s}" for ph in masks.values()))
        for t in tests:
            print(
                f"   {t[:30]:30s}"
                + "".join(f"{d[ph][t][0]:12.1f} ({d[ph][t][1]:4.2f})" if t in d[ph] else f"{'-':>20s}" for ph in masks.values())
            )
