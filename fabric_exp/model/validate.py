#!/usr/bin/env python3
"""Check the static model against the configurations measured on hardware (fabric_exp/RESULTS.md).

usage: python3 fabric_exp/model/validate.py [-v]
"""
import copy
import dataclasses
import sys

from fabric_model import Options, baseline_placement, evaluate, load_catalog, make_cores

# (label, scenario, overrides to the scenario's rx block, options, measured cycles/packet, source)
CASES = [
    ("speedy neighbor, 1 KiB", "nbr_speedy", {}, Options(payload=1024), 123.6, "§10"),
    ("speedy neighbor, 2 KiB", "nbr_speedy", {}, Options(payload=2048), 123.7, "§10"),
    ("speedy neighbor, 4 KiB", "nbr_speedy", {}, Options(payload=4096), 124.5, "§10"),
    ("speedy neighbor, 5.4 KB", "nbr_speedy", {}, Options(payload=5440), 154.2, "§10"),
    ("speedy neighbor, 7.6 KB", "nbr_speedy", {}, Options(payload=7616), 213.1, "§10"),
    ("speedy neighbor, 8.7 KB", "nbr_speedy", {}, Options(payload=8704), 243.9, "§10"),
    ("2D 1-hop E/W write", "2d_single_ew", {}, Options(), 223.0, "§8"),
    ("2D 1-hop E/W write, trimmed to 2 ch", "2d_single_ew", {}, Options(sender_channels=2), 168.1, "§8"),
    (
        "2D 1-hop E/W scatter, trimmed to 2 ch",
        "2d_single_ew",
        {"local": "scatter"},
        Options(sender_channels=2),
        191.0,
        "§8",
    ),
    ("2D 1-hop N/S write", "2d_single_ns", {}, Options(), 223.1, "§8"),
    ("2D 1-hop N/S write, trimmed to 1 ch", "2d_single_ns", {}, Options(sender_channels=1), 152.4, "§8"),
    (
        "2D 1-hop N/S scatter, trimmed to 1 ch",
        "2d_single_ns",
        {"local": "scatter"},
        Options(sender_channels=1),
        206.4,
        "§8",
    ),
    ("2D 3-hop middle router, trimmed", "2d_3hop_middle", {}, Options(sender_channels=2), 188.7, "§8"),
    ("2D mesh multicast, heaviest link", "2d_mcast_heavy_link", {}, Options(), 180.6, "§8"),
    # 4 KiB (results/d4k.csv; cycles = 1.35e9 / packets per second)
    ("2D 1-hop E/W write, 4 KiB", "2d_single_ew", {}, Options(payload=4096), 221.6, "§8"),
    ("2D 1-hop E/W scatter, 4 KiB", "2d_single_ew", {"local": "scatter"}, Options(payload=4096), 220.9, "§8"),
    ("2D 1-hop N/S write, 4 KiB", "2d_single_ns", {}, Options(payload=4096), 222.0, "§8"),
    ("2D 1-hop N/S scatter, 4 KiB", "2d_single_ns", {"local": "scatter"}, Options(payload=4096), 221.3, "§8"),
    ("2D 3-hop write, 4 KiB", "2d_3hop_middle", {}, Options(payload=4096), 221.2, "§8"),
    ("2D mesh multicast, heaviest link, 4 KiB", "2d_mcast_heavy_link", {}, Options(payload=4096), 181.9, "§8"),
    # One-way at 4 KiB with the keepalive at 1000 (TT_FABRIC_EXP_SEQ_UPDATE_TIMEOUT, the catalog default): the routers
    # are the limit again (results/ka1000_*.csv)
    (
        "2D 1-hop E/W write, 4 KiB, trimmed to 2 ch",
        "2d_single_ew",
        {},
        Options(payload=4096, sender_channels=2),
        169.8,
        "§16",
    ),
    (
        "2D 1-hop E/W scatter, 4 KiB, trimmed to 2 ch",
        "2d_single_ew",
        {"local": "scatter"},
        Options(payload=4096, sender_channels=2),
        191.1,
        "§16",
    ),
    (
        "2D 1-hop N/S write, 4 KiB, trimmed to 1 ch",
        "2d_single_ns",
        {},
        Options(payload=4096, sender_channels=1),
        152.5,
        "§16",
    ),
    (
        "2D 1-hop N/S scatter, 4 KiB, trimmed to 1 ch",
        "2d_single_ns",
        {"local": "scatter"},
        Options(payload=4096, sender_channels=1),
        206.5,
        "§16",
    ),
    (
        "2D 3-hop middle router, 4 KiB, trimmed",
        "2d_3hop_middle",
        {},
        Options(payload=4096, sender_channels=2),
        186.5,
        "§16",
    ),
    ("speedy neighbor one way, 2 KiB", "nbr_speedy_oneway", {}, Options(payload=2048), 122.8, "§16"),
    ("speedy neighbor one way, 4000 B", "nbr_speedy_oneway", {}, Options(payload=4000), 122.9, "§16"),
    ("speedy neighbor one way, 4 KiB", "nbr_speedy_oneway", {}, Options(payload=4096), 122.9, "§16"),
]

# The same one-way configurations at the firmware's keepalive (8000 cycles): the floor the keepalive sets (§15-16).
KEEPALIVE = 8000
FIRMWARE_KEEPALIVE_CASES = [
    ("2D 1-hop E/W write, 4 KiB, trimmed", "2d_single_ew", {}, Options(payload=4096, sender_channels=2), 187.6, "§15"),
    (
        "2D 1-hop E/W scatter, 4 KiB, trimmed",
        "2d_single_ew",
        {"local": "scatter"},
        Options(payload=4096, sender_channels=2),
        191.1,
        "§15",
    ),
    ("2D 1-hop N/S write, 4 KiB, trimmed", "2d_single_ns", {}, Options(payload=4096, sender_channels=1), 187.6, "§15"),
    (
        "2D 1-hop N/S scatter, 4 KiB, trimmed",
        "2d_single_ns",
        {"local": "scatter"},
        Options(payload=4096, sender_channels=1),
        206.5,
        "§15",
    ),
    (
        "2D 3-hop middle router, 4 KiB, trimmed",
        "2d_3hop_middle",
        {},
        Options(payload=4096, sender_channels=2),
        196.4,
        "§15",
    ),
    ("2D 1-hop E/W write, 2 KiB, trimmed", "2d_single_ew", {}, Options(payload=2048, sender_channels=2), 168.1, "§8"),
    ("2D 1-hop N/S write, 2 KiB, trimmed", "2d_single_ns", {}, Options(payload=2048, sender_channels=1), 152.4, "§8"),
    ("speedy neighbor one way, 2 KiB", "nbr_speedy_oneway", {}, Options(payload=2048), 129.6, "§15"),
    ("speedy neighbor one way, 4000 B", "nbr_speedy_oneway", {}, Options(payload=4000), 130.3, "§15"),
    ("speedy neighbor one way, 4 KiB", "nbr_speedy_oneway", {}, Options(payload=4096), 187.5, "§15"),
]


def run(cases, base_cat, cores, verbose, keepalive=None):
    print(f"{'configuration':46s} {'measured':>8s} {'model':>7s} {'error':>6s}  bottleneck   per-core load")
    errs = []
    for label, scen, rx_over, opts, measured, src in cases:
        cat = copy.deepcopy(base_cat)
        cat["scenarios"][scen]["rx"].update(rx_over)
        if keepalive is not None:
            opts = dataclasses.replace(opts, keepalive=keepalive)
        r = evaluate(cat, scen, baseline_placement(cat["scenarios"][scen]), cores, opts)
        err = 100 * (r.interval / measured - 1)
        errs.append(abs(err))
        loads = "  ".join(f"{c}={v:.0f}" for c, v in r.loads.items())
        print(
            f"{label:46s} {measured:8.1f} {r.interval:7.1f} {err:+5.1f}%  {r.bottleneck:10s}   {loads} (wire {r.wire:.0f})"
        )
        if verbose:
            for core, items in r.breakdown.items():
                print(f"      {core}: " + ", ".join(f"{k} {v:.0f}" for k, v in items.items()))
    print(f"\nmean |error| {sum(errs) / len(errs):.1f}%, max {max(errs):.1f}%")


def main():
    verbose = "-v" in sys.argv
    base_cat = load_catalog()
    cores = make_cores()
    run(CASES, base_cat, cores, verbose)
    print(f"\nOne-way benchmarks at the firmware keepalive ({KEEPALIVE} cycles):")
    run(FIRMWARE_KEEPALIVE_CASES, base_cat, cores, verbose, keepalive=KEEPALIVE)


if __name__ == "__main__":
    main()
