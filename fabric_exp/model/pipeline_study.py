#!/usr/bin/env python3
"""Receive-side pipelines for two ERISCs + one DRISC: who decodes, who admits (downstream credits), who moves data.

Designs (each keeps all-or-nothing admission and per-receiver-channel order):
  today   E1 decodes, admits and moves data; a blocked packet is decoded again on every retry
  P0      as today, but a blocked packet keeps its decoded action (retry = credit check only); no DRISC
  P3      an ERISC decodes once and admits; the admitted packet is pushed to the DRISC with its downstream slots,
          and the DRISC moves the data (credits never leave the Ethernet core)
  P3s     P3 with the DRISC doing local delivery too, forwards on one NoC and local delivery on the other
  P1      the packet is pushed to the DRISC first; the DRISC decodes and sends the output list back; the ERISC admits
          and grants; the DRISC moves the data. The Ethernet slot frees right after the push
  P2      as P1, but the DRISC admits against credits the ERISC relays to it (no per-packet round trip)
  P4      the ERISC decodes once and does local delivery; packets with forwards are pushed to the DRISC, which admits
          and forwards, against credits the ERISC relays (P4) or that downstream routers return to it directly (P4d,
          a protocol change)
  P5      push: the ERISC pushes every packet to a DRISC that sends on both NoCs; the DRISC decodes, admits against
          credits the ERISC relays (whatever has arrived at each relay), and moves the data, forwards on one NoC and local delivery on the other.
          The ERISC polls, pushes, relays credits and retires Ethernet slots
  P5L     as P5, but the ERISC also decodes and does local delivery, in parallel with the DRISC's forwards (P4 with
          the forward/local NoC split)

Handoff and DRISC costs are the measured lean-issue values (catalog.yaml, RESULTS §14).

usage: python3 fabric_exp/model/pipeline_study.py [designs|congestion|sensitivity|sharing|scan|alpha ...]
       [--scenario NAME ...]
"""
import copy
import sys

from des import simulate
from fabric_model import Options, describe, evaluate, load_catalog, make_cores, placements, wire_cycles

PAYLOAD = 4096
LEVERS = {"idle_poll": "lean", "b_ptr": 4, "b_ack": 4, "k_drain": 4}
SCENARIOS = ["2d_3hop_middle", "2d_mcast_ns_router"]
SIM = dict(sender_slots=32, receiver_slots=32, cycles=250000, warmup=50000)
STAGING = 12  # DRISC staging slots per router (P1 needs ~12: its slots are held across the round trip)

# (DRISCs, fixed pieces, extra options)
DESIGNS = {
    "today": (
        0,
        {
            "tx_send": "E0",
            "tx_completion": "E0",
            "rx_poll": "E1",
            "rx_decode": "E1",
            "rx_admit": "E1",
            "rx_local": "E1",
            "rx_forward": "E1",
            "rx_ack": "E1",
        },
        {},
    ),
    "P0 keep decode": (
        0,
        {
            "tx_send": "E0",
            "tx_completion": "E0",
            "rx_poll": "E1",
            "rx_decode": "E1",
            "rx_admit": "E1",
            "rx_local": "E1",
            "rx_forward": "E1",
            "rx_ack": "E1",
        },
        {"cache_decode": True},
    ),
    "P3 E admits, D moves": (1, {"rx_decode": "E*", "rx_admit": "E*", "rx_forward": "D0"}, {"cache_decode": True}),
    "P3s P3, fwd/local NoCs": (
        1,
        {"rx_decode": "E*", "rx_admit": "E*", "rx_local": "D0", "rx_forward": "D0"},
        {"cache_decode": True, "drisc_split": True},
    ),
    "P1 D decodes, E admits": (1, {"rx_decode": "D0", "rx_admit": "E*", "rx_forward": "D0"}, {}),
    "P2 credit relay": (1, {"rx_decode": "D0", "rx_admit": "D0", "rx_forward": "D0"}, {}),
    "P4 D admits, relay": (
        1,
        {"rx_decode": "E*", "rx_local": "E*", "rx_admit": "D0", "rx_forward": "D0"},
        {"cache_decode": True},
    ),
    "P4d D admits, own credits": (
        1,
        {"rx_decode": "E*", "rx_local": "E*", "rx_admit": "D0", "rx_forward": "D0"},
        {"cache_decode": True, "drisc_credits_direct": True},
    ),
    "P5 push, D does the rest": (
        1,
        {"rx_decode": "D0", "rx_admit": "D0", "rx_local": "D0", "rx_forward": "D0"},
        {"drisc_split": True},
    ),
    "P5L push, E local": (
        1,
        {"rx_decode": "E*", "rx_local": "E*", "rx_admit": "D0", "rx_forward": "D0"},
        {"cache_decode": True, "drisc_split": True},
    ),
}
DRISC_DESIGNS = [d for d in DESIGNS if DESIGNS[d][0]]


def matches(pl, fixed):
    for piece, core in fixed.items():
        if piece not in pl:
            continue
        if core == "E*":
            if not pl[piece].startswith("E"):
                return False
        elif pl[piece] != core:
            return False
    return True


def best(cat, scen, design, opts_over=None, extra_fixed=None):
    """Best static placement for a design, ties broken toward balanced loads."""
    nd, fixed, extra = DESIGNS[design]
    fixed = dict(fixed, **(extra_fixed or {}))
    cores = make_cores(nd)
    opts = Options(payload=PAYLOAD, **{**LEVERS, **extra, **(opts_over or {})})
    found = []
    for pl in placements(cat, scen, cores):
        if not matches(pl, fixed):
            continue
        r = evaluate(cat, scen, pl, cores, opts)
        if r.valid:
            found.append((r.interval, sorted(r.loads.values(), reverse=True), pl, r))
    found.sort(key=lambda x: (x[0], x[1]))
    _, _, pl, r = found[0]
    return pl, r, cores, opts


def loads(r):
    return " ".join(f"{c}={v:.0f}" for c, v in sorted(r.loads.items()))


def section_designs(cat, scenarios):
    line = wire_cycles(cat, PAYLOAD)
    print(
        f"1. Designs at {PAYLOAD} B with lean poll + batch x4 (line rate <= {line:.0f} cycles/packet). Event-driven: "
        f"32/32 Ethernet slots, {STAGING} DRISC staging slots, downstream channels of 16 and 24 slots.\n"
    )
    for scen in scenarios:
        print(f"=== {scen}")
        print(
            f"  {'design':24s} {'static':>6s} {'ds16':>5s} {'ds24':>5s} {'retry':>5s}  {'static per-core load':24s} "
            f"placement"
        )
        for design in DESIGNS:
            pl, r, cores, opts = best(cat, scen, design)
            ev = []
            for dsl in (16, 24):
                st = {}
                iv, _ = simulate(
                    cat, scen, pl, cores, opts, staging_slots=STAGING, downstream={"slots": dsl}, stats=st, **SIM
                )
                ev.append((iv, st["A"]["retries_per_pkt"]))
            print(
                f"  {design:24s} {r.interval:6.0f} {ev[0][0]:5.0f} {ev[1][0]:5.0f} {ev[1][1]:5.2f}  "
                f"{loads(r):24s} {describe(pl)}"
            )
        print()


def section_congestion(cat, scenarios):
    print("2. Downstream congestion on router A only, with E1 carrying transmit work (tx_completion on E1).")
    print("   A<-B: packets A receives and forwards (the congested path); A->B: A's own transmit direction.")
    print("   Event-driven, 32/32 Ethernet slots, 24-slot downstream channels.\n")
    cases = {
        "none": None,
        "drain x1.5": {"factor": 1.5, "congested": ("A",)},
        "stalls 1500/4000": {"stall": (4000, 1500), "congested": ("A",)},
    }
    for scen in scenarios:
        print(f"=== {scen}")
        print(f"  {'design':24s} " + "  ".join(f"{c + ': A<-B  A->B retry':>34s}" for c in cases))
        for design in DESIGNS:
            if design == "today":  # today's placement keeps all transmit work on E0
                continue
            pl, r, cores, opts = best(cat, scen, design, extra_fixed={"tx_completion": "E1"})
            row = []
            for dsd in cases.values():
                d = dict(dsd or {}, slots=24)
                st = {}
                simulate(cat, scen, pl, cores, opts, staging_slots=STAGING, downstream=d, stats=st, **SIM)
                row.append(f"{st['A']['interval']:5.0f} {st['B']['interval']:5.0f} {st['A']['retries_per_pkt']:5.2f}")
            print(f"  {design:24s} " + "  ".join(f"{x:>34s}" for x in row))
        # the same with today's re-decode on retry, E1 carrying completions
        pl, r, cores, opts = best(
            cat, scen, "today", opts_over={"cache_decode": False}, extra_fixed={"tx_completion": "E1"}
        )
        row = []
        for dsd in cases.values():
            d = dict(dsd or {}, slots=24)
            st = {}
            simulate(cat, scen, pl, cores, opts, staging_slots=STAGING, downstream=d, stats=st, **SIM)
            row.append(f"{st['A']['interval']:5.0f} {st['B']['interval']:5.0f} {st['A']['retries_per_pkt']:5.2f}")
        print(f"  {'(re-decode on retry)':24s} " + "  ".join(f"{x:>34s}" for x in row))
        print()


def with_costs(cat, dispatch=None, alpha=None, stateless=False):
    c = copy.deepcopy(cat)
    if dispatch is not None:  # CPU per forward copy on the DRISC
        c["core_types"]["drisc"]["forward_per_copy"] = dispatch
    if alpha is not None:
        c["core_types"]["drisc"]["alpha"] = alpha
    if stateless:  # stock stateless NoC calls instead of lean stateful issue (RESULTS §14, port free, net of timer)
        k = c["connectors"]
        k["erisc_to_drisc_payload"].update(producer=26, producer_per_batch=20)  # write 21 + doorbell 20 (push 46)
        k["erisc_to_drisc_header"].update(producer=21, producer_per_batch=20)
        k["drisc_to_erisc"]["producer_per_batch"] = 24  # DRISC doorbell 24
        k["drisc_to_erisc_request"]["producer_per_batch"] = 45
        k["erisc_to_drisc_grant"]["producer_per_batch"] = 20
        k["credit_relay"]["producer_per_batch"] = 20
        c["core_types"]["drisc"]["forward_per_copy"] = (dispatch or 45) + 14  # write 21 + doorbell 24 vs lean 31
    return c


SENS_DESIGNS = ["P0 keep decode", "P3 E admits, D moves", "P5 push, D does the rest", "P5L push, E local"]


def section_sensitivity(cat, scenarios):
    print("3. Sensitivity to the costs that are still estimated or depend on implementation (static model, best")
    print("   placement per design). Defaults: lean stateful issue, DRISC forward copy 45 (31 measured + 14 estimated")
    print("   for the next-hop slot address and header update), alpha 1.0 (measured).\n")
    variants = {
        "defaults": {},
        "forward copy 31 (no header work)": {"dispatch": 31},
        "forward copy 60": {"dispatch": 60},
        "stateless issue (stock NoC calls)": {"stateless": True},
        "DRISC alpha 1.2": {"alpha": 1.2},
    }
    for scen in scenarios:
        print(f"=== {scen}")
        print(f"  {'variant':36s} " + " ".join(f"{d.split()[0]:>6s}" for d in SENS_DESIGNS) + "   bottleneck (P5)")
        for vname, v in variants.items():
            c = with_costs(cat, **v)
            res = [best(c, scen, d) for d in SENS_DESIGNS]
            print(
                f"  {vname:36s} " + " ".join(f"{r.interval:6.0f}" for _, r, _, _ in res) + f"   {res[2][1].bottleneck}"
            )
        print()


def section_sharing(cat, scenarios):
    print(
        "4. One DRISC serving two routers (E/W routers sharing the two spare dual-NoC DRISCs): the DRISC carries both"
    )
    print("   routers' work and NoC traffic. Event-driven, 24-slot downstream, 12 staging slots.\n")
    for scen in scenarios:
        print(f"=== {scen}")
        print(f"  {'design':26s} {'own DRISC: static event  E1  D0':>34s}   {'shared by 2: static event  E1  D0':>36s}")
        for design in ("P5 push, D does the rest", "P5L push, E local"):
            row = []
            for share in (1, 2):
                pl, r, cores, opts = best(cat, scen, design, opts_over={"drisc_share": share})
                iv, _ = simulate(cat, scen, pl, cores, opts, staging_slots=STAGING, downstream={"slots": 24}, **SIM)
                row.append(f"{r.interval:6.0f} {iv:6.0f} {r.loads['E1']:4.0f} {r.loads['D0']:4.0f}")
            print(f"  {design:26s} {row[0]:>34s}   {row[1]:>36s}")
        print()


def section_scan(cat, scenarios):
    print("5. Moving E0's sender-channel scan (and connection checks) to E1: E1 finds packets and hands each to E0")
    print(
        "   through their shared L1 (descriptor + doorbell); E0 only sends. Static model (the event-driven model does"
    )
    print("   not model this). 4 and 9 sender channels.\n")
    for scen in scenarios:
        print(f"=== {scen}")
        print(
            f"  {'design':26s} {'channels':>8s}  {'E0 scans: interval  E0  E1  D0':>32s}   "
            f"{'E1 scans: interval  E0  E1  D0':>32s}"
        )
        for design in ("today", "P5 push, D does the rest", "P5L push, E local"):
            for ch in (4, 9):
                row = []
                for scan_on in (None, "E1"):
                    pl, r, cores, opts = best(
                        cat, scen, design, opts_over={"sender_channels": ch, "tx_scan_on": scan_on}
                    )
                    d0 = f"{r.loads['D0']:4.0f}" if "D0" in r.loads else "   -"
                    row.append(f"{r.interval:6.0f} {r.loads['E0']:4.0f} {r.loads['E1']:4.0f} {d0}")
                print(f"  {design:26s} {ch:8d}  {row[0]:>32s}   {row[1]:>32s}")
        print()


def section_alpha(cat, scenarios):
    print("6. DRISC speed. alpha = DRISC cycles per ERISC cycle for the same work, applied to all of its work; our")
    print("   DRISC microbenchmark measures 1.0 (RESULTS §14), so this is sensitivity only. Event-driven, 24-slot")
    print("   downstream, 12 staging slots; DRISC forward copy 45 / 60 before scaling.\n")
    alphas = (1.0, 1.25, 1.5)
    for scen in scenarios:
        print(f"=== {scen}")
        _, r0, _, _ = best(cat, scen, "P0 keep decode")
        print(f"  no DRISC (P0): static {r0.interval:.0f}")
        print(f"  {'design':26s} " + "  ".join(f"{'alpha ' + str(a) + ': 45   60':>20s}" for a in alphas))
        for design in DRISC_DESIGNS:
            row = []
            for a in alphas:
                ivs = []
                for per_copy in (None, 60):
                    c = with_costs(cat, dispatch=per_copy, alpha=a)
                    pl, r, cores, opts = best(c, scen, design)
                    iv, _ = simulate(c, scen, pl, cores, opts, staging_slots=STAGING, downstream={"slots": 24}, **SIM)
                    ivs.append(iv)
                row.append(f"{ivs[0]:5.0f} {ivs[1]:5.0f}")
            print(f"  {design:26s} " + "  ".join(f"{x:>20s}" for x in row))
        print()


def main():
    args = sys.argv[1:]
    scenarios = [a for i, a in enumerate(args) if i > 0 and args[i - 1] == "--scenario"] or SCENARIOS
    sections = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--scenario")]
    cat = load_catalog()
    for s in sections or ["designs", "congestion", "sensitivity", "sharing", "scan", "alpha"]:
        {
            "designs": section_designs,
            "congestion": section_congestion,
            "sensitivity": section_sensitivity,
            "sharing": section_sharing,
            "scan": section_scan,
            "alpha": section_alpha,
        }[s](cat, scenarios)


if __name__ == "__main__":
    main()
