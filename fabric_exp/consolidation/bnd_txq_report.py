#!/usr/bin/env python3
"""bnd_txq_report.py TAG [ROW ...]: the TX-queue probe (variants/v4bdq, router_txq_probe.hpp; bnd_txq_runs.sh) per row
and sending ERISC (E0 rows of results/bndq_<TAG>_<row>_phase.csv with sampled sends), cycles at 1.35 GHz:
  rate       the row's cycles/packet with the probe (results/bndq_<TAG>_<row>.csv)
  sends      61 x sampled sends (the probe samples every 61st send)
  busy%      normal sends whose post-send status read (the post-send group) saw the queue busy
  polls      busy polls per such send (the spin before the packets-received update)
  data       sampled: cycles from the data command until a poll reads the queue free (includes >= 1 poll)
  poll       sampled: one poll on the then idle queue (the poll's own cost); data - poll ~ the queue's busy time
  dmax       sampled: largest data wait
  reg        sampled: the same after the packets-received register command
  hist       sampled data waits in 16-cycle buckets (%), the last open (>= 112)
  yields     pre-send yields (queue busy before a send) per 1000 sends"""
import csv, glob, os, re, sys

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
EVENTS = ["tx_blk_credit", "tx_blk_txq", "rx_blk_admit", "rx_blk_trid"]
PHASES = ["tx_send", "tx_txq_spin", "tx_completion", "tx_conn", "rx_decode_admit", "rx_forward", "rx_local",
          "rx_completion", "rx_ack", "tx_txq_prespin", "tx_visit", "rx_visit"]
SLOTS = EVENTS + [p + "_ticks" for p in PHASES] + [p + "_count" for p in PHASES]


def rate(lab):
    f = f"{R}/{lab}.csv"
    if not os.path.exists(f):
        return float("nan")
    v = [1.35e9 / float(r["avg_packets_per_s"]) for r in csv.DictReader(open(f))]
    return sum(v) / len(v)


def main():
    tag = sys.argv[1]
    rows = sys.argv[2:] or sorted(re.sub(rf"^bndq_{tag}_|_phase\.csv$", "", os.path.basename(f))
                                  for f in glob.glob(f"{R}/bndq_{tag}_*_phase.csv"))
    print(f"{'row':16}{'erisc':>14}{'rate':>7}{'sends':>11}{'busy%':>7}{'polls':>6}{'data':>7}{'poll':>6}{'dmax':>6}"
          f"{'reg':>6}{'yield/k':>8}  hist 0-15,16-31,...,112+")
    for row in rows:
        lab = f"bndq_{tag}_{row}"
        f = f"{R}/{lab}_phase.csv"
        if not os.path.exists(f):
            continue
        cpp = rate(lab)
        tg = []
        for r in csv.DictReader(open(f)):
            s = [int(r[k] or 0) for k in SLOTS]
            if s[4] == 0:
                continue
            tg.append((s[4], r, s))
        for n, r, s in sorted(tg, key=lambda x: -x[0])[:3]:
            sends = 61 * n
            normal = sends - n
            busy = 100.0 * s[1] / normal if normal else 0
            polls = s[2] / s[1] if s[1] else 0
            hist = ",".join(f"{100.0 * s[10 + b] / n:.0f}" for b in range(8))
            who = f"{r['device'][-7:-1]}/{r['eth_channel']}"
            print(f"{row[:16]:16}{who:>14}{cpp:7.1f}{sends:11d}{busy:7.1f}{polls:6.2f}{s[5] / n:7.1f}{s[9] / n:6.1f}"
                  f"{s[6]:6d}{s[7] / n:6.1f}{1000.0 * s[3] / sends:8.2f}  {hist}")


if __name__ == "__main__":
    main()
