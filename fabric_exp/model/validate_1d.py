#!/usr/bin/env python3
"""Validate the static model on 1D linear multicast, router by router (RESULTS §1, §2).

The 1D benchmarks multicast all-to-all along lines of 4 chips (the 2x4 mesh as two lines, plus the links between
them), so routers carry different traffic: an end chip receives three flows and sends one, its neighbour sends three
and receives one, a middle link carries two each way. The phase profile (results/p2_phase.csv) gives each router's
sent and received packet counts and what each received packet needs (local delivery, a forward). Each router becomes
a scenario with tx_rate / rx_rate = its packets per packet the test reports; the slowest router sets the predicted
interval, compared with the reported one (results/notel.csv, no telemetry). Delay injection (RESULTS §2) is emulated
with Options(tx_delay / rx_delay). Section 3 runs the receive-pipeline designs on the same per-router traffic.

usage: python3 fabric_exp/model/validate_1d.py [validate|designs]
"""
import copy
import csv
import os
from collections import defaultdict

import sys

from fabric_model import Options, baseline_placement, evaluate, load_catalog, make_cores

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
CLOCK = 1.35e9
TESTS = [
    ("LinearMulticast", "write"),
    ("LinearMulticast", "scatter"),
    ("UnidirLinearMulticast", "write"),
    ("UnidirLinearMulticast", "scatter"),
]
SIZES = (1024, 2048, 4096)
DECODE_1D = 28  # decode + admission check, measured 26-30 on every 1D router (p2_phase.csv)
# RESULTS §2 (2 KiB): change in the interval with TX +30, TX +60, RX +30, RX +60 cycles per packet.
DELAY = {
    ("LinearMulticast", "write"): (28, 52, 42, 71),
    ("LinearMulticast", "scatter"): (0, 16, 45, 74),
    ("UnidirLinearMulticast", "write"): (0, 13, 45, 75),
    ("UnidirLinearMulticast", "scatter"): (0, 0, 45, 75),
}
# The injected spin (fabric_experiment_spin) polls the cycle counter: it costs N plus the first timer read (~7) plus up
# to one ~10-cycle loop iteration of overshoot, ~N + 12. On the sender it overlaps the ~14 cycles the send already
# waits on the Ethernet TX queue (RESULTS §3), so it adds ~N - 2 there.
SPIN_OVERHEAD = 12
TXQ_WAIT = 14


def measured(test, kind, size):
    """(test cycles, reported packets, reported interval) from the no-telemetry summary."""
    ntype = "NOC_UNICAST_SCATTER_WRITE" if kind == "scatter" else "NOC_UNICAST_WRITE"
    for r in csv.DictReader(open(os.path.join(RESULTS, "notel.csv"))):
        if r["test_name"] == test and r["ntype"] == ntype and int(r["packet_size"]) == size:
            t, pps = float(r["avg_cycles"]), float(r["avg_packets_per_s"])
            return t, t * pps / CLOCK, CLOCK / pps
    raise KeyError((test, kind, size))


def routers(test, kind, size):
    """Per router (device, Ethernet channel): sent and received packets, forwards and local deliveries per received."""
    name = f"{test}_size_{size}_ntype_unicast_{'scatter_' if kind == 'scatter' else ''}write_num_links_1"
    out = defaultdict(lambda: {"tx": 0.0, "rx": 0.0, "fwd": 0.0, "local": 0.0})
    for r in csv.DictReader(open(os.path.join(RESULTS, "p2_phase.csv"))):
        if r["test"] != name:
            continue
        key = (r["device"].replace("MeshCoordinate", ""), r["eth_channel"])
        p = float(r["pkts"] or 0)
        if r["erisc"] == "0":
            out[key]["tx"] = p
        elif p:
            out[key].update(rx=p, fwd=float(r["rx_forward_count"]) / p, local=float(r["rx_local_count"]) / p)
    return out


def router_scenario(r, kind, n, per_flow):
    return {
        "path": "generic",
        "header_bytes": 48,  # 1D packet header
        "credit_bytes": 32,  # 1D counter credits
        "sender_channels": 2,  # 1D linear: worker + upstream router
        "active_channels": min(2, max(1, round(r["tx"] / per_flow))),  # one per flow up to the two channels
        "pkts_per_iter": 1.0,
        "rx": {"decode": DECODE_1D, "local": kind if r["local"] > 0.5 else "none", "forwards": round(r["fwd"])},
        "tx_rate": r["tx"] / n,
        "rx_rate": r["rx"] / n,
    }


def router_catalogs(cat, test, kind, size):
    """One catalog copy per router, with that router as scenario "_router"."""
    _, n, _ = measured(test, kind, size)
    rs = routers(test, kind, size)
    per_flow = min(r["tx"] for r in rs.values() if r["tx"])
    for key, r in rs.items():
        c = copy.deepcopy(cat)
        c["scenarios"]["_router"] = router_scenario(r, kind, n, per_flow)
        yield key, r, c


def predict(cat, test, kind, size, opts):
    """Predicted interval per reported packet: the slowest router, with its bottleneck."""
    best = None
    for key, r, c in router_catalogs(cat, test, kind, size):
        res = evaluate(c, "_router", baseline_placement(c["scenarios"]["_router"]), make_cores(0), opts)
        if best is None or res.interval > best[0]:
            best = (res.interval, key, r, res)
    return best


def section_validate(cat):
    print("1. Interval per reported packet (cycles): the slowest router in the model vs the benchmark (no telemetry).")
    print(f"   Decode + admission {DECODE_1D}; other piece costs from the 2D-calibrated catalog.\n")
    print(f"  {'test':32s} {'size':>5s} {'measured':>8s} {'model':>6s} {'error':>6s}   slowest router in the model")
    errs = []
    for test, kind in TESTS:
        for size in SIZES:
            _, _, m = measured(test, kind, size)
            iv, key, r, res = predict(cat, test, kind, size, Options(payload=size))
            errs.append(abs(iv / m - 1))
            loads = " ".join(f"{c}={v:.0f}" for c, v in sorted(res.loads.items()))
            print(
                f"  {test + ' ' + kind:32s} {size:5d} {m:8.1f} {iv:6.1f} {100 * (iv / m - 1):+5.1f}%   "
                f"{key[0]} ch {key[1]}: sends {int(r['tx'])}, receives {int(r['rx'])} "
                f"(fwd {r['fwd']:.0f}, local {r['local']:.0f}); {res.bottleneck}, {loads}"
            )
    print(f"\n  mean |error| {100 * sum(errs) / len(errs):.1f}%, max {100 * max(errs):.1f}%\n")

    print(
        "2. Delay injection at 2 KiB (RESULTS §2): change in the interval, measured / model. The model adds the spin's"
    )
    print(
        f"   own cost: N + {SPIN_OVERHEAD} on the receiver, N + {SPIN_OVERHEAD} - {TXQ_WAIT} on the sender (overlapping its TX-queue wait).\n"
    )
    print(f"  {'test':32s} {'TX +30':>11s} {'TX +60':>11s} {'RX +30':>11s} {'RX +60':>11s}")
    tx_eff = lambda n: n + SPIN_OVERHEAD - TXQ_WAIT
    rx_eff = lambda n: n + SPIN_OVERHEAD
    for test, kind in TESTS:
        base = predict(cat, test, kind, 2048, Options(payload=2048))[0]
        cells = []
        for i, opts in enumerate(
            [
                Options(payload=2048, tx_delay=tx_eff(30)),
                Options(payload=2048, tx_delay=tx_eff(60)),
                Options(payload=2048, rx_delay=rx_eff(30)),
                Options(payload=2048, rx_delay=rx_eff(60)),
            ]
        ):
            d = predict(cat, test, kind, 2048, opts)[0] - base
            cells.append(f"{DELAY[(test, kind)][i]:+4d} / {d:+4.0f}")
        print(f"  {test + ' ' + kind:32s} " + " ".join(f"{c:>11s}" for c in cells))


def section_designs(cat):
    from pipeline_study import DESIGNS, LEVERS, best

    designs = [
        "today",
        "P3 E admits, D moves",
        "P3s P3, fwd/local NoCs",
        "P5 push, D does the rest",
        "P5L push, E local",
    ]
    print("3. Receive-pipeline designs on the measured 1D traffic, 4 KiB, lean poll + batch x4 (static; the")
    print(
        "   event-driven model assumes symmetric traffic). Per design: the slowest router's interval per reported packet"
    )
    print("   (line rate <= 117), that router, and its per-core loads.\n")
    for test, kind in TESTS:
        print(f"=== {test} {kind}")
        for design in designs:
            worst = None
            for key, r, c in router_catalogs(cat, test, kind, 4096):
                pl, res, cores, opts = best(c, "_router", design)
                if worst is None or res.interval > worst[0]:
                    worst = (res.interval, key, r, res)
            iv, key, r, res = worst
            loads = " ".join(f"{cn}={v:.0f}" for cn, v in sorted(res.loads.items()))
            print(
                f"  {design:26s} {iv:6.0f}  {key[0]} ch {key[1]} (sends {int(r['tx'])}, receives {int(r['rx'])}, "
                f"fwd {r['fwd']:.0f}): {res.bottleneck}, {loads}"
            )
        print()


def main():
    cat = load_catalog()
    sections = sys.argv[1:] or ["validate", "designs"]
    for sec in sections:
        {"validate": section_validate, "designs": section_designs}[sec](cat)


if __name__ == "__main__":
    main()
