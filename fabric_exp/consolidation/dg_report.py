#!/usr/bin/env python3
"""dg_report.py LABEL...: the completion-credit counters of the diagnostic builds v4gidg / v4vcdg (router_diag.hpp), per
test, from results/<label>_phase.csv (TT_FABRIC_TXQ_COUNTERS=1), with the rate from results/<label>.csv.

For each test it shows the busiest sending ERISC (E0 with the most worker slots credited back) and the busiest
receiving ERISC (E1 with the most slots credited), all per packet unless named otherwise:
  E0  credits      credit_worker calls per packet, and the mean slots per call
      sizes        share of calls returning 1, 2, 3, 4, 5+ slots
      cyc/call     cycles inside credit_worker per call
      blocked      passes where the worker channel had a packet and the far ring too few free slots
  E1  credits      completion credits per packet, and the mean slots per credit
      n=1..4       share of credits of 1..4 slots
      idle         share of credits made on a pass without a packet
      unal         share of credits whose first slot is not group-aligned
      frames       credit frames sent per packet; put off: frames delayed by a busy TX queue
      proof        cycles per proof attempt; failed: share of attempts that failed
      idlepart     passes without a packet while a partial group waited
"""
import csv, os, sys

RES = "/home/aagarwal/tt-metal/fabric_exp/results"
EVENTS = ["tx_blk_credit", "tx_blk_txq", "rx_blk_admit", "rx_blk_trid"]
PHASES = ["tx_send", "tx_txq_spin", "tx_completion", "tx_conn", "rx_decode_admit", "rx_forward", "rx_local",
          "rx_completion", "rx_ack", "tx_txq_prespin", "tx_visit", "rx_visit"]
SLOTS = EVENTS + [p + "_ticks" for p in PHASES] + [p + "_count" for p in PHASES]  # router_diag.hpp's slot k


def slot(row, k):
    return int(row[SLOTS[k]])


def per(a, b):
    return a / b if b else 0.0


def rates(label):
    p = os.path.join(RES, label + ".csv")
    if not os.path.exists(p):
        return []
    return [(r["test_name"], r["packet_size"], 1.35e9 / float(r["avg_packets_per_s"])) for r in csv.DictReader(open(p))]


def tests(label):
    order, rows = [], {}
    for r in csv.DictReader(open(os.path.join(RES, label + "_phase.csv"))):
        if r["test"] not in rows:
            order.append(r["test"])
            rows[r["test"]] = []
        rows[r["test"]].append(r)
    return [(t, rows[t]) for t in order]


def e0_line(r):
    pk = int(r["pkts"])
    calls = slot(r, 7)
    sizes = [per(slot(r, k), calls) for k in range(1, 6)]
    return (f"E0 {r['device']:>8} ch{r['eth_channel']:>2}  pkts {pk:9d}  credits {per(calls, pk):.3f} (x{per(pk, calls):.2f})"
            f"  sizes {' '.join(f'{s:.2f}' for s in sizes)}  cyc/call {per(slot(r, 6), calls):5.1f}"
            f"  blocked {per(slot(r, 0), pk):.3f}")


def e1_line(r):
    pk = int(r["pkts"])
    credits = sum(slot(r, k) for k in range(16))
    by_n = [sum(slot(r, (n - 1) + 4 * i + 8 * u) for i in (0, 1) for u in (0, 1)) for n in range(1, 5)]
    idle = sum(slot(r, k) for k in range(16) if k & 4)
    unal = sum(slot(r, k) for k in range(8, 16))
    attempts = credits + slot(r, 18)
    return (f"E1 {r['device']:>8} ch{r['eth_channel']:>2}  pkts {pk:9d}  credits {per(credits, pk):.3f} (x{per(pk, credits):.2f})"
            f"  n=1..4 {' '.join(f'{per(c, credits):.2f}' for c in by_n)}  idle {per(idle, credits):.2f}"
            f"  unal {per(unal, credits):.2f}  frames {per(slot(r, 16), pk):.3f} put off {per(slot(r, 17), pk):.3f}"
            f"  proof {per(slot(r, 19), attempts):5.1f} failed {per(slot(r, 18), attempts):.3f}"
            f"  idlepart {per(slot(r, 20), pk):.3f}")


for label in sys.argv[1:]:
    print(f"== {label}")
    rs = rates(label)
    for i, (t, rows) in enumerate(tests(label)):
        rate = f"{rs[i][0]} {rs[i][1]} B {rs[i][2]:.2f} cycles/packet" if i < len(rs) else "no rate"
        print(f"{t}: {rate}")
        e0 = [r for r in rows if r["erisc"] == "0" and int(r["pkts"]) > 0]
        e1 = [r for r in rows if r["erisc"] == "1" and int(r["pkts"]) > 0]
        if e0:
            print("  " + e0_line(max(e0, key=lambda r: int(r["pkts"]))))
        if e1:
            print("  " + e1_line(max(e1, key=lambda r: int(r["pkts"]))))
