#!/usr/bin/env python3
"""Search placements and levers with the static model.

For each scenario, core set and lever set it finds the placement with the smallest per-packet interval at the given
payload (default 4 KiB) and checks it against line rate (the target: 400G at 4 KiB, i.e. the router keeps up with
the wire).

usage: python3 fabric_exp/model/explore.py [scenario ...] [--payload N] [--detail]   # static search
       python3 fabric_exp/model/explore.py [scenario ...] [--payload N] --verify     # best placements, event-driven
       python3 fabric_exp/model/explore.py [scenario ...] [--payload N] --slots      # event-driven vs slot depth
"""
import copy
import sys

from fabric_model import (
    Options,
    baseline_placement,
    describe,
    evaluate,
    load_catalog,
    make_cores,
    search,
    slots_needed,
    wire_cycles,
)

# "lean poll": idle sender channels get only the packet check (no completion check without packets in flight, lazy
# connection check); no protocol change. polling_study.py compares it with fused polling and a summary doorbell.
LEVERS = {
    "today": {},
    "lean poll": {"idle_poll": "lean"},
    "batch x4": {"b_ptr": 4, "b_ack": 4, "k_drain": 4},
    "lean + batch x2": {"idle_poll": "lean", "b_ptr": 2, "b_ack": 2, "k_drain": 2},
    "lean + batch x4": {"idle_poll": "lean", "b_ptr": 4, "b_ack": 4, "k_drain": 4},
    "lean + batch x4 + 2 NoCs": {"idle_poll": "lean", "b_ptr": 4, "b_ack": 4, "k_drain": 4, "noc_ports": 2},
}
# Core sets: (DRISCs, catalog overrides for the DRISC). BH gives Metal 16 DRISCs for at most 10 active Ethernet cores,
# so each router can have its own DRISC (drisc_study.py also shows a DRISC shared by two routers).
ARCHS = {"2 ERISC": (0, {}), "2 ERISC + DRISC": (1, {})}


def with_drisc(cat, overrides):
    """Catalog copy with the DRISC core type's fields overridden (alpha, share, egress_bytes_per_cycle, loop)."""
    if not overrides:
        return cat
    c = copy.deepcopy(cat)
    c["core_types"]["drisc"].update(overrides)
    return c


def parse_args():
    argv = sys.argv[1:]
    payload = 4096
    if "--payload" in argv:
        i = argv.index("--payload")
        payload = int(argv[i + 1])
        del argv[i : i + 2]
    return [a for a in argv if not a.startswith("--")] or None, payload


def main():
    cat = load_catalog()
    scenarios, payload = parse_args()
    detail = "--detail" in sys.argv
    line = wire_cycles(cat, payload)
    print(
        f"{payload} B payload: line rate needs <= {line:.0f} cycles/packet "
        f"({cat['clock_hz'] * payload / line / 1e9 * 8:.0f} Gb/s of payload)\n"
    )
    for scen in scenarios or list(cat["scenarios"]):
        print(f"=== {scen}")
        base = evaluate(cat, scen, baseline_placement(cat["scenarios"][scen]), make_cores(), Options(payload=payload))
        print(
            f"  today's placement: {base.interval:.0f} cycles/packet "
            f"({'  '.join(f'{c}={v:.0f}' for c, v in base.loads.items())})"
        )
        print(
            f"  {'levers':22s} {'cores':16s} {'cyc/pkt':>7s} {'GB/s':>5s} {'line':>4s} {'limit':>8s} {'slots':>5s}  "
            f"best placement"
        )
        for lname, lev in LEVERS.items():
            for aname, (nd, over) in ARCHS.items():
                cores, acat = make_cores(nd), with_drisc(cat, over)
                opts = Options(payload=payload, **lev)
                pl, r = search(acat, scen, cores, opts, top=1)[0]
                ok = "yes" if r.interval <= line + 0.5 else "-"
                print(
                    f"  {lname:22s} {aname:16s} {r.interval:7.0f} {r.gbps(cat, payload):5.1f} {ok:>4s} "
                    f"{r.bottleneck:>8s} {slots_needed(cat, scen, opts, r.interval):5d}  {describe(pl)}"
                )
                if detail:
                    for core, items in r.breakdown.items():
                        print(
                            f"        {core} ({r.loads[core]:.0f}): "
                            + ", ".join(f"{k} {v:.0f}" for k, v in items.items())
                        )
        print()


def verify(scenarios=None, top=3, payload=4096):
    """Re-run the static search's best placements in the event-driven model."""
    from des import simulate

    cat = load_catalog()
    for scen in scenarios or list(cat["scenarios"]):
        slots = 32 if cat["scenarios"][scen]["path"] == "speedy" else 16
        print(f"=== {scen} ({payload} B, {slots} slots)")
        print(f"  {'levers':22s} {'cores':22s} {'static':>6s} {'event':>6s}  placement")
        for lname, lev in LEVERS.items():
            for aname, (nd, over) in ARCHS.items():
                cores, acat = make_cores(nd), with_drisc(cat, over)
                opts = Options(payload=payload, **lev)
                for pl, r in search(acat, scen, cores, opts, top=top if nd else 1):
                    iv, util = simulate(
                        acat,
                        scen,
                        pl,
                        cores,
                        opts,
                        sender_slots=slots,
                        receiver_slots=slots,
                        cycles=250000,
                        warmup=50000,
                    )
                    print(f"  {lname:22s} {aname:22s} {r.interval:6.0f} {iv:6.0f}  {describe(pl)}")
        print()


def slot_sweep(scenarios=None, payload=4096, depths=(8, 16, 24, 32, 48)):
    """Event-driven interval of the best 2-ERISC placement per lever as the sender/receiver slot count grows."""
    from des import simulate

    cat = load_catalog()
    cores = make_cores()
    print(f"{'scenario':20s} {'levers':22s} {'static':>6s}  " + " ".join(f"{d:>5d}" for d in depths))
    for scen in scenarios or ["nbr_speedy", "2d_single_ew", "2d_single_ns"]:
        for lname, lev in LEVERS.items():
            opts = Options(payload=payload, **lev)
            pl, r = search(cat, scen, cores, opts, top=1)[0]
            ivs = [
                simulate(cat, scen, pl, cores, opts, sender_slots=d, receiver_slots=d, cycles=250000, warmup=50000)[0]
                for d in depths
            ]
            print(f"{scen:20s} {lname:22s} {r.interval:6.0f}  " + " ".join(f"{iv:5.0f}" for iv in ivs))


if __name__ == "__main__":
    if "--slots" in sys.argv:
        names, size = parse_args()
        slot_sweep(names, payload=size)
    elif "--verify" in sys.argv:
        names, size = parse_args()
        verify(names, payload=size)
    else:
        main()
