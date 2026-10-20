#!/usr/bin/env python3
"""Two ERISCs plus one DRISC: where the DRISC pays off, and how much that depends on its unmeasured costs.

1. gain:        best placement with and without the DRISC, per 2D scenario and lever, static and event-driven
2. sensitivity: how the gain moves with the DRISC's estimated costs (push cost, speed, sharing, NoC port rate), and
                the break-even push cost and speed
3. staging:     DRISC staging slots the best DRISC placements need (event-driven)

usage: python3 fabric_exp/model/drisc_study.py [gain|sensitivity|staging ...] [--scenario NAME ...] [--payload N]

Default payload 4 KiB: the first target is 400G at 4 KiB, i.e. the router keeps up with the wire (~117 cycles/packet).
"""
import copy
import sys

from des import simulate
from explore import LEVERS, with_drisc
from fabric_model import Options, describe, load_catalog, make_cores, search, wire_cycles

SCENARIOS = [
    "2d_single_ew",
    "2d_single_ns",
    "2d_single_ew_scatter",
    "2d_single_ns_scatter",
    "2d_3hop_middle",
    "2d_mcast_heavy_link",
    "2d_mcast_ns_router",
]
STATIC_ONLY = {"2d_mcast_heavy_link"}  # several active sender channels; the event-driven model feeds one
SIM = dict(cycles=250000, warmup=50000)
MIN_GAIN = 0.05  # a DRISC placement must beat the best two-ERISC placement by this much to count
PAYLOAD = 4096


def best(cat, scen, num_drisc, lev):
    return search(cat, scen, make_cores(num_drisc), Options(payload=PAYLOAD, **lev), top=1)[0]


def event(cat, scen, pl, num_drisc, lev, slots, **kw):
    if scen in STATIC_ONLY:
        return None
    iv, _ = simulate(
        cat,
        scen,
        pl,
        make_cores(num_drisc),
        Options(payload=PAYLOAD, **lev),
        sender_slots=slots,
        receiver_slots=slots,
        **SIM,
        **kw,
    )
    return iv


def gain(a, b):
    """Relative reduction in cycles/packet from a to b."""
    return 1 - b / a


def fmt(x, w=5):
    return f"{x:{w}.0f}" if x is not None else f"{'-':>{w}s}"


def uses_drisc(pl):
    return any(c.startswith("D") for c in pl.values())


def drisc_part(pl):
    return describe(pl) if uses_drisc(pl) else "(DRISC unused)"


def section_gain(cat, scenarios):
    print(
        f"1. Best placement without and with one DRISC ({PAYLOAD} B; cycles/packet; event-driven at 16 and 32 "
        f"slots; line rate <= {wire_cycles(cat, PAYLOAD):.0f})\n"
    )
    for scen in scenarios:
        print(f"=== {scen}" + ("  (event-driven n/a: several active channels)" if scen in STATIC_ONLY else ""))
        print(f"  {'levers':22s} {'2 ERISC':>17s}   {'2 ERISC + DRISC':>17s}   {'gain':>11s}  placement with DRISC")
        print(
            f"  {'':22s} {'static':>6s} {'16':>5s} {'32':>5s}   {'static':>6s} {'16':>5s} {'32':>5s}   "
            f"{'static':>6s} {'32':>4s}"
        )
        for lname, lev in LEVERS.items():
            pl0, r0 = best(cat, scen, 0, lev)
            pl1, r1 = best(cat, scen, 1, lev)
            e0 = [event(cat, scen, pl0, 0, lev, s) for s in (16, 32)]
            e1 = [event(cat, scen, pl1, 1, lev, s) for s in (16, 32)]
            g32 = f"{100 * gain(e0[1], e1[1]):3.0f}%" if e0[1] else "   -"
            print(
                f"  {lname:22s} {fmt(r0.interval, 6)} {fmt(e0[0])} {fmt(e0[1])}   {fmt(r1.interval, 6)} "
                f"{fmt(e1[0])} {fmt(e1[1])}   {100 * gain(r0.interval, r1.interval):5.0f}% {g32}  {drisc_part(pl1)}"
            )
        print()


def drisc_gain(cat, scen, lev):
    r0 = best(cat, scen, 0, lev)[1]
    pl1, r1 = best(cat, scen, 1, lev)
    return gain(r0.interval, r1.interval), pl1


def with_push(cat, producer):
    c = copy.deepcopy(cat)
    c["connectors"]["erisc_to_drisc_payload"]["producer"] = producer
    c["connectors"]["erisc_to_drisc_header"]["producer"] = min(
        producer, c["connectors"]["erisc_to_drisc_header"]["producer"]
    )
    return c


def break_even(values, cat_for, scen, lev):
    """Largest value in `values` for which the DRISC still gains at least MIN_GAIN (None if never)."""
    last = None
    for v in values:
        if drisc_gain(cat_for(v), scen, lev)[0] >= MIN_GAIN:
            last = v
        else:
            break
    return last


def section_sensitivity(cat, scenarios):
    print(f"2. Sensitivity of the DRISC gain to its estimated costs (static model, {PAYLOAD} B)\n")
    print("  Defaults: push 30 cycles + 24 per doorbell batch on the ERISC, alpha 1 (DRISC as fast as an ERISC),")
    print(
        f"  one DRISC per router, DRISC NoC port 54 B/cycle. Break-even = last value still gaining >= "
        f"{100 * MIN_GAIN:.0f}%.\n"
    )
    push_steps = list(range(30, 205, 5))
    alpha_steps = [round(1 + 0.05 * i, 2) for i in range(0, 41)]
    print(
        f"  {'scenario':20s} {'levers':22s} {'gain':>5s} {'shared':>6s} {'27B/cy':>6s} "
        f"{'push x2':>7s} {'alpha 1.5':>9s}   {'break-even push':>15s} {'alpha':>6s}"
    )
    for scen in scenarios:
        for lname, lev in LEVERS.items():
            g, _ = drisc_gain(cat, scen, lev)
            if g < MIN_GAIN:
                print(f"  {scen:20s} {lname:22s} {100 * g:4.0f}%   (DRISC does not help)")
                continue
            shared = drisc_gain(with_drisc(cat, {"share": 2}), scen, lev)[0]
            slow_port = drisc_gain(with_drisc(cat, {"egress_bytes_per_cycle": 27}), scen, lev)[0]
            push2 = drisc_gain(with_push(cat, 60), scen, lev)[0]
            a15 = drisc_gain(with_drisc(cat, {"alpha": 1.5}), scen, lev)[0]
            be_push = break_even(push_steps, lambda v: with_push(cat, v), scen, lev)
            be_alpha = break_even(alpha_steps, lambda v: with_drisc(cat, {"alpha": v}), scen, lev)
            be_push_s = f"{be_push}" + ("+" if be_push == push_steps[-1] else "")
            be_alpha_s = f"{be_alpha}" + ("+" if be_alpha == alpha_steps[-1] else "")
            print(
                f"  {scen:20s} {lname:22s} {100 * g:4.0f}% {100 * shared:5.0f}% {100 * slow_port:5.0f}% "
                f"{100 * push2:6.0f}% {100 * a15:8.0f}%   {be_push_s:>15s} {be_alpha_s:>6s}"
            )
    print()


def section_staging(cat, scenarios, depths=(1, 2, 4, 8, 16)):
    print(f"3. DRISC staging slots (event-driven, {PAYLOAD} B, 32 Ethernet slots; NoC latency 150 each way)\n")
    print(
        f"  {'scenario':20s} {'levers':22s} {'2 ERISC':>7s}  " + " ".join(f"{d:>5d}" for d in depths) + "   placement"
    )
    for scen in scenarios:
        if scen in STATIC_ONLY:
            continue
        for lname, lev in LEVERS.items():
            pl1, _ = best(cat, scen, 1, lev)
            if not uses_drisc(pl1):
                continue
            pl0, _ = best(cat, scen, 0, lev)
            base = event(cat, scen, pl0, 0, lev, 32)
            ivs = [event(cat, scen, pl1, 1, lev, 32, staging_slots=d) for d in depths]
            print(
                f"  {scen:20s} {lname:22s} {base:7.0f}  " + " ".join(f"{iv:5.0f}" for iv in ivs) + f"   {describe(pl1)}"
            )
    print()


def main():
    global PAYLOAD
    args = sys.argv[1:]
    if "--payload" in args:
        i = args.index("--payload")
        PAYLOAD = int(args[i + 1])
        del args[i : i + 2]
    scenarios = [a for i, a in enumerate(args) if i > 0 and args[i - 1] == "--scenario"] or SCENARIOS
    sections = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--scenario")]
    sections = sections or ["gain", "sensitivity", "staging"]
    cat = load_catalog()
    for s in sections:
        {"gain": section_gain, "sensitivity": section_sensitivity, "staging": section_staging}[s](cat, scenarios)


if __name__ == "__main__":
    main()
