#!/usr/bin/env python3
"""Sender-channel polling on the sending ERISC (E0): what each scheme costs, and who pays for it.

Every scheme still checks every channel for new packets; they differ in what else a visit checks and in how the
checks are issued:
  scan            today: per channel, a packet check, a completion check and a connection/teardown check
  lean            per channel only the packet check; completions only on channels with packets in flight; connection
                  and teardown checked one visit in 16. No protocol change.
  fused           lean, with every channel's packet check in one block of reads and one branch (reads block)
  fused_overlap   fused, if the core keeps the reads in flight together (unmeasured)
  summary         a shared counter every producer also increments; idle channels are skipped while it agrees with the
                  active channels' counts. Each producer pays one more inline write per packet (charged here to the
                  forwarding core; a worker pays it for its own packets, which this model does not show)

usage: python3 fabric_exp/model/polling_study.py [--payload N]
"""
import sys

from des import simulate
from fabric_model import Options, baseline_placement, evaluate, load_catalog, make_cores, wire_cycles

PAYLOAD = 4096
SCHEMES = {
    "scan (today)": {},
    "lean": {"idle_poll": "lean"},
    "fused": {"idle_poll": "fused"},
    "fused_overlap": {"idle_poll": "fused_overlap"},
    "summary": {"summary_doorbell": True},
    "summary + lean": {"summary_doorbell": True, "idle_poll": "lean"},
}
BATCH = {"no batching": {}, "batch x4": {"b_ptr": 4, "b_ack": 4, "k_drain": 4}}
CASES = [  # (label, scenario, sender channels)
    ("1-hop E/W, VC0 (4 ch)", "2d_single_ew", 4),
    ("1-hop E/W, VC0+VC1 (9 ch)", "2d_single_ew", 9),
    ("3-hop middle, VC0 (4 ch)", "2d_3hop_middle", 4),
    ("mcast heavy link (3 of 4 active)", "2d_mcast_heavy_link", 4),
]
SIM = dict(sender_slots=32, receiver_slots=32, cycles=250000, warmup=50000)


def main():
    global PAYLOAD
    args = sys.argv[1:]
    if "--payload" in args:
        PAYLOAD = int(args[args.index("--payload") + 1])
    cat = load_catalog()
    cores = make_cores(0)
    print(
        f"Sender-channel polling schemes, {PAYLOAD} B, today's placement (E0 sends, E1 receives); line rate <= "
        f"{wire_cycles(cat, PAYLOAD):.0f} cycles/packet."
    )
    print(
        "E0 = sending ERISC's cycles per packet (poll = its per-packet share of loop + polling); E1 = receiving "
        "ERISC; event = event-driven interval at 32/32 slots.\n"
    )
    for label, scen, channels in CASES:
        print(f"=== {label}")
        print(f"  {'scheme':16s} " + "   ".join(f"{b:>27s}" for b in BATCH))
        print(f"  {'':16s} " + "   ".join(f"{'E0  poll    E1 static event':>27s}" for _ in BATCH))
        for sname, s in SCHEMES.items():
            row = []
            for b in BATCH.values():
                opts = Options(payload=PAYLOAD, sender_channels=channels, **s, **b)
                pl = baseline_placement(cat["scenarios"][scen])
                r = evaluate(cat, scen, pl, cores, opts)
                poll = sum(v for k, v in r.breakdown["E0"].items() if k.startswith("loop + polls"))
                ev = "    -"
                if cat["scenarios"][scen]["active_channels"] == 1:
                    iv, _ = simulate(cat, scen, pl, cores, opts, **SIM)
                    ev = f"{iv:5.0f}"
                row.append(f"{r.loads['E0']:4.0f} {poll:5.0f} {r.loads['E1']:5.0f} {r.interval:6.0f} {ev}")
            print(f"  {sname:16s} " + "   ".join(f"{x:>27s}" for x in row))
        print()


if __name__ == "__main__":
    main()
