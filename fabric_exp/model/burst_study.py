#!/usr/bin/env python3
"""Bursty traffic: what batching is worth when packets do not stream non-stop.

Batching in the model is opportunistic: a sender visit sends whatever is queued (up to k_drain), one Ethernet pointer
update covers the packets sent in that visit (up to b_ptr), and the receiver acks what it has retired (up to b_ack)
as soon as it runs out of new packets. Nothing waits for a batch to fill. So batching only acts when packets queue,
which depends on the burst length and on how fast the producer writes packets compared with the router.

Each worker starts a burst of B packets every `period` cycles; burst time = burst start to the last packet's local
delivery on the far side (includes NoC and Ethernet latency, ~1000 cycles for a lone packet).

usage: python3 fabric_exp/model/burst_study.py [--scenario NAME] [--payload N]
"""
import sys

from des import simulate
from fabric_model import Options, baseline_placement, load_catalog, make_cores, wire_cycles

PAYLOAD = 4096
SCHEMES = {
    "today": {},
    "batch x4": {"b_ptr": 4, "b_ack": 4, "k_drain": 4},
    "lean": {"idle_poll": "lean"},
    "lean + batch x4": {"idle_poll": "lean", "b_ptr": 4, "b_ack": 4, "k_drain": 4},
}
BURSTS = (1, 2, 4, 8, 16, 32, 64)
WORKER = (40, 150)  # worker cycles per packet: benchmark producer; a producer slower than the batched router


def run(cat, scen, opts, size, per_pkt):
    period = size * 500 + 6000
    st = {}
    simulate(
        cat,
        scen,
        baseline_placement(cat["scenarios"][scen]),
        make_cores(0),
        opts,
        sender_slots=32,
        receiver_slots=32,
        cycles=period * 30,
        warmup=period * 4,
        stats=st,
        burst=(size, period, per_pkt),
    )
    a = st["A"]
    return a["burst_time"], a["pkts_per_send_visit"]


def main():
    args = sys.argv[1:]
    scen = args[args.index("--scenario") + 1] if "--scenario" in args else "2d_single_ew"
    payload = int(args[args.index("--payload") + 1]) if "--payload" in args else PAYLOAD
    cat = load_catalog()
    print(
        f"Bursty traffic, {scen}, {payload} B, 32/32 slots, today's placement (line rate <= "
        f"{wire_cycles(cat, payload):.0f} cycles/packet)."
    )
    print("Per cell: burst time / B (cycles per packet, latency included), and packets per sender visit.\n")
    for per_pkt in WORKER:
        print(f"=== worker writes one packet per {per_pkt} cycles")
        print(f"  {'B':>3s}  " + "  ".join(f"{s:>16s}" for s in SCHEMES) + "   batching gain (lean -> lean + batch)")
        for size in BURSTS:
            res = {s: run(cat, scen, Options(payload=payload, **o), size, per_pkt) for s, o in SCHEMES.items()}
            cells = "  ".join(f"{t / size:9.0f} ({v:3.1f})" for t, v in res.values())
            gain = 1 - res["lean + batch x4"][0] / res["lean"][0]
            print(f"  {size:3d}  {cells}   {100 * gain:5.1f}%")
        print()


if __name__ == "__main__":
    main()
