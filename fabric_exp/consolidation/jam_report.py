#!/usr/bin/env python3
"""jam_report.py: the ring-jam counters (v4txa11j, RING_JAM_FIX.md §8) per test and ring axis, and the decision rule.

  jam_report.py SYSTEM CSV...      SYSTEM: galaxy | p150. CSVs: TT_FABRIC_PHASE_PROFILE_CSV files of the on-arm.

The CSV columns keep the phase profile's names (RING_JAM_FIX.md §8): busy = contests, blocked = contests with < 2 free
far slots, idle = passes with the in-ring channel holding a packet, gap = in-ring sends, tx_blk_credit = in-ring
far-credit refusals, tx_blk_txq = injection refusals, rx_blk_admit = injection sends, <phase>_count = contests with i
free far slots (12 phases, i = 0..10, 11 = 11+), <phase>_ticks = contests with an in-ring backlog of i.
Rows are E0 rows whose info carries "jam". Axis by the CSV's direction (RoutingDirection: 0 N, 1 E, 2 S, 3 W): Galaxy 4x8
(ring / XY torus) E/W =
the 4-long X axis (K 2), N/S = the 8-long Y axis (K 6); P150 1D ring: every router on the 8-ring (K 6). R = K + 2.
Per (test, axis): contests, the share with room >= R, the share with room < 2, in-ring refusals per in-ring send,
the median in-ring backlog at contests, contests per pass."""
import collections
import csv
import sys

PHASES = ["tx_send", "tx_txq_spin", "tx_completion", "tx_conn", "rx_decode_admit", "rx_forward", "rx_local",
          "rx_completion", "rx_ack", "tx_txq_prespin", "tx_visit", "rx_visit"]


def axis_k(system, direction):
    if system == "galaxy":
        return ("X (4-long)", 2) if direction in (1, 3) else ("Y (8-long)", 6)
    return ("8-ring", 6)


def median_bucket(hist):
    n = sum(hist)
    if not n:
        return None
    acc = 0
    for i, v in enumerate(hist):
        acc += v
        if acc * 2 >= n:
            return i if i < 11 else "11+"


def main():
    system, files = sys.argv[1], sys.argv[2:]
    agg = collections.defaultdict(lambda: collections.Counter())
    hists = collections.defaultdict(lambda: {"room": [0] * 12, "backlog": [0] * 12})
    for f in files:
        for r in csv.DictReader(open(f)):
            if r["erisc"] != "0" or (int(r["info"]) >> 40) != 0x6A616D:
                continue
            d = int(r["direction"])
            ax, k = axis_k(system, d)
            key = (f.rsplit("/", 1)[-1], r["test"], ax, k)
            a = agg[key]
            for col, name in (("iters", "passes"), ("pkts", "sends"), ("busy", "contests"), ("blocked", "room_lt2"),
                              ("idle", "in_ring_ready"), ("gap", "in_ring_sends"), ("tx_blk_credit", "in_ring_refused"),
                              ("tx_blk_txq", "inj_refused"), ("rx_blk_admit", "inj_sends")):
                a[name] += int(r[col])
            a["routers"] += 1
            for i, p in enumerate(PHASES):
                hists[key]["room"][i] += int(r[f"{p}_count"])
                hists[key]["backlog"][i] += int(r[f"{p}_ticks"])
    print("| file | test | axis | R | routers | contests | contests/pass | room>=R | room<2 | in-ring refusals/send | "
          "median backlog | inj/in-ring sends |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key in sorted(agg):
        f, test, ax, k = key
        a, h = agg[key], hists[key]
        R = k + 2
        n = a["contests"]
        ge = sum(h["room"][min(R, 11):]) if R < 11 else h["room"][11]
        pct = lambda x: f"{100.0 * x / n:.0f}%" if n else "-"
        print(f"| {f} | {test} | {ax} | {R} | {a['routers']} | {n} | {n / a['passes'] if a['passes'] else 0:.2f} | "
              f"{pct(ge)} | {pct(a['room_lt2'])} | "
              f"{a['in_ring_refused'] / a['in_ring_sends'] if a['in_ring_sends'] else 0:.2f} | "
              f"{median_bucket(h['backlog'])} | {a['inj_sends'] / a['in_ring_sends'] if a['in_ring_sends'] else 0:.2f} |")


if __name__ == "__main__":
    main()
