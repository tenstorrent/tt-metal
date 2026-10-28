#!/usr/bin/env python3
"""rt_report.py LABEL...: the worker credit loop's round-trip breakdown from the timing builds v4girt / v4vcrt
(router_rt.hpp), per test, from results/<label>_phase.csv (TT_FABRIC_TXQ_COUNTERS=1), with the rate from
results/<label>.csv. All times in ERISC cycles (1.35 GHz), means over the sampled packets (E0: every 61st worker
packet; E1: one slot at a time, ~1 in 64; the two sample different packets, so `rest` is a difference of means):
  E0 (sending chip, the busiest worker channel)
      send->seen   from the send to E0 seeing its completion: link out, all of E1, the frame, link back, E0's next look
      queued       packets left in the worker channel at the send
      hist         share of send->seen in 256-cycle buckets (the last one open)
  E1 (receiving chip, the busiest receiver channel 0)
      head->ret    at the head with its data present -> retired (its writes issued)
      ret->cred    -> its group credited (batching wait and proof); idle / partial: share and mean of the samples
                   credited on a pass without a packet / in a credit of fewer than T slots
      cred->frame  -> the frame carrying it sent
      present      packets present when it reached the head (counting it)
      pos          its mean position in its credit (0 = oldest)
  rest = send->seen - (head->ret + ret->cred + cred->frame): link both ways, E1's wait before the packet reaches the
         head, and E0's look at the count after the frame lands.
"""
import csv, os, sys

RES = "/home/aagarwal/tt-metal/fabric_exp/results"
EVENTS = ["tx_blk_credit", "tx_blk_txq", "rx_blk_admit", "rx_blk_trid"]
PHASES = ["tx_send", "tx_txq_spin", "tx_completion", "tx_conn", "rx_decode_admit", "rx_forward", "rx_local",
          "rx_completion", "rx_ack", "tx_txq_prespin", "tx_visit", "rx_visit"]
SLOTS = EVENTS + [p + "_ticks" for p in PHASES] + [p + "_count" for p in PHASES]  # router_rt.hpp's slot k


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


def summary(rows):
    """(E0 numbers, E1 numbers) of a test's busiest worker channel and receiver, or None where there is none."""
    e0 = [r for r in rows if r["erisc"] == "0" and slot(r, 0) > 0]
    e1 = [r for r in rows if r["erisc"] == "1" and slot(r, 0) > 0]
    a = b = None
    if e0:
        r = max(e0, key=lambda r: slot(r, 0))
        n = slot(r, 0)
        a = dict(n=n, seen=per(slot(r, 1), n), queued=per(slot(r, 3), n), max=slot(r, 4),
                 hist=[per(slot(r, k), n) for k in range(5, 13)])
    if e1:
        r = max(e1, key=lambda r: slot(r, 0))
        n = slot(r, 0)
        b = dict(n=n, h2r=per(slot(r, 1), n), r2c=per(slot(r, 2), n), c2f=per(slot(r, 3), n),
                 present=per(slot(r, 4), n), idle=per(slot(r, 5), n), idle_r2c=per(slot(r, 6), slot(r, 5)),
                 partial=per(slot(r, 7), n), partial_r2c=per(slot(r, 8), slot(r, 7)), pos=per(slot(r, 9), n),
                 hist=[per(slot(r, k), n) for k in range(10, 18)])
    return a, b


if __name__ == "__main__":
    for label in sys.argv[1:]:
        print(f"== {label}")
        rs = rates(label)
        for i, (t, rows) in enumerate(tests(label)):
            rate = f"{rs[i][0]} {rs[i][1]} B {rs[i][2]:.2f} cycles/packet" if i < len(rs) else "no rate"
            print(f"{t}: {rate}")
            a, b = summary(rows)
            if a:
                print(f"  E0 n {a['n']:6d}  send->seen {a['seen']:7.1f} (max {a['max']:6d})"
                      f"  queued {a['queued']:5.2f}  hist/256 {' '.join(f'{x:.2f}' for x in a['hist'])}")
            if b:
                print(f"  E1 n {b['n']:6d}  head->ret {b['h2r']:6.1f}  ret->cred {b['r2c']:6.1f}"
                      f" (idle {b['idle']:.2f} @{b['idle_r2c']:6.1f}, partial {b['partial']:.2f} @{b['partial_r2c']:6.1f})"
                      f"  cred->frame {b['c2f']:5.1f}  present {b['present']:5.2f}  pos {b['pos']:4.2f}"
                      f"  hist/128 {' '.join(f'{x:.2f}' for x in b['hist'])}")
            if a and b:
                print(f"  rest {a['seen'] - b['h2r'] - b['r2c'] - b['c2f']:7.1f}")
