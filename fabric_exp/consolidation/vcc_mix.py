#!/usr/bin/env python3
"""vcc_mix.py PREFIX ALLOC... : the mixed VC0 + VC1 load rows (vc1/vc1_mix.yaml) per allocation, from
results/<PREFIX>_<alloc>_p1_o3_13129088_vc1_tests.csv (pass 2 averaged in when present). For each mixed test and size:
the VC0 flow's and the VC1 flow's cycles/packet (1.35 GHz / packets per second; the VC1 flow is the row from device 0),
the VC0 flow alone (VC1Solo*), and `overlap`: the shorter flow's duration over the longer's (1.0 = they ran together
the whole time; below that the longer one ran alone at the end, so its rate is a little flattered)."""
import csv, os, sys
from collections import defaultdict

RES = "/home/aagarwal/tt-metal/fabric_exp/results"


def load(prefix, alloc):
    out = defaultdict(list)  # (test, size, device) -> [(cycles/packet, cycles)]
    for p in (1, 2):
        f = os.path.join(RES, f"{prefix}_{alloc}_p{p}_{os.environ.get('VCA_OPT', 'o3')}_13129088_vc1_tests.csv")
        if not os.path.exists(f):
            continue
        for r in csv.DictReader(open(f)):
            test = r["test_name"].split("_size_")[0]
            out[(test, r["packet_size"], r["device"])].append(
                (1.35e9 / float(r["packets_per_second"]), int(r["cycles"])))
    return {k: (sum(x[0] for x in v) / len(v), sum(x[1] for x in v) / len(v)) for k, v in out.items()}


def main():
    prefix, allocs = sys.argv[1], sys.argv[2:]
    print(f"{'allocation':10} {'load':8} {'size':>5}  {'VC0 alone':>9} {'VC0 mixed':>9} {'VC1 mixed':>9} {'overlap':>7}")
    for a in allocs:
        d = load(prefix, a)
        for kind, vc0_dev in (("Worker", "1"), ("Forward", "3")):
            for size in ("16", "4352"):
                solo = d.get((f"VC1Solo{kind}", size, vc0_dev))
                vc0 = d.get((f"VC1Mix{kind}", size, vc0_dev))
                vc1 = d.get((f"VC1Mix{kind}", size, "0"))
                if not (vc0 and vc1):
                    continue
                overlap = min(vc0[1], vc1[1]) / max(vc0[1], vc1[1])
                print(f"{a:10} {kind:8} {size:>5}  {solo[0] if solo else float('nan'):9.1f} {vc0[0]:9.1f}"
                      f" {vc1[0]:9.1f} {overlap:7.2f}")


if __name__ == "__main__":
    main()
