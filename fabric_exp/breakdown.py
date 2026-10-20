#!/usr/bin/env python3
"""Compose a per-packet ERISC cycle budget for each test from the isolated-timer runs.

Each phase's per-occurrence cost comes from the run that timed only that phase, minus the calibrated cycles a timer
pair shows inside the phase it brackets. Occurrence counts per packet come from the counters-only run. The total is
the profiler-off interval of the busiest link (the bandwidth summary's cycles/packet). For the ERISC that is the
bottleneck, the unattributed remainder is loop and polling overhead; for the other ERISC it also includes idle time.

usage: breakdown.py [results_dir]   (expects runs d0, m_cnt, m_iter, m_send, m_spin, m_compl, m_conn, m_rx, m_visit)
"""
import csv
import sys
from collections import defaultdict

RES = sys.argv[1] if len(sys.argv) > 1 else "fabric_exp/results"
PHASE_RUN = {
    "tx_send": "m_send",
    "tx_txq_spin": "m_spin",
    "tx_completion": "m_compl",
    "tx_conn": "m_conn",
    "rx_decode_admit": "m_rx",
    "rx_forward": "m_rx",
    "rx_local": "m_rx",
    "rx_completion": "m_rx",
    "rx_ack": "m_rx",
    "tx_visit": "m_visit",
    "rx_visit": "m_visit",
}


def load_phase(label):
    rows = {}
    for r in csv.DictReader(open(f"{RES}/{label}_phase.csv")):
        rows[(r["test"], r["device"], r["eth_channel"], r["erisc"])] = r
    return rows


def load_interval(label):
    out = {}
    for r in csv.DictReader(open(f"{RES}/{label}.csv")):
        out[(r["test_name"], r["ntype"].replace("NOC_UNICAST_", "").lower())] = (
            float(r["avg_bandwidth_gigabytes_per_s"]),
            1.35e9 / float(r["avg_packets_per_s"]),
        )
    return out


def interval_for(test, intervals):
    for (name, ntype), v in intervals.items():
        if test.startswith(name + "_") and f"ntype_unicast_{ntype}" in test:
            return v
    return None


def cost(rows, key, phase):
    """Per-occurrence cycles of `phase` at core `key`, corrected for the timer's in-phase cost."""
    r = rows.get(key)
    if not r or int(r[phase + "_count"]) == 0:
        return None
    cal_inner = (int(r["timer_cal"]) & 0xFFFFFFFF) / 100.0
    return int(r[phase + "_ticks"]) / int(r[phase + "_count"]) - cal_inner


def main():
    runs = {label: load_phase(label) for label in set(PHASE_RUN.values()) | {"m_cnt", "m_iter"}}
    base = load_interval("d0")
    delay = {k: load_interval(k) for k in ("m_tx30", "m_rx30")}
    counts = runs["m_cnt"]
    by_test = defaultdict(list)
    for key, r in counts.items():
        by_test[key[0]].append((key, r))

    for test, cores in by_test.items():
        iv = interval_for(test, base)
        tx30 = interval_for(test, delay["m_tx30"])
        rx30 = interval_for(test, delay["m_rx30"])
        print(
            f"\n=== {test}: {iv[0]:.2f} GB/s, {iv[1]:.1f} cycles/packet on the busiest link"
            f" (TX+30 -> {tx30[1] - iv[1]:+.0f}, RX+30 -> {rx30[1] - iv[1]:+.0f})"
        )
        for erisc, role in (("0", "TX"), ("1", "RX")):
            sel = [(k, r) for k, r in cores if k[3] == erisc and int(r["pkts"]) > 0]
            top = max(int(r["pkts"]) for _, r in sel)
            busiest = [(k, r) for k, r in sel if int(r["pkts"]) == top]
            key, r = busiest[0]
            pk = int(r["pkts"])
            it = runs["m_iter"].get(key)
            util = ""
            if it:
                act = int(it["busy"]) + int(it["blocked"]) + int(it["idle"])
                util = (
                    f"busy {100 * int(it['busy']) / act:.0f}% blocked {100 * int(it['blocked']) / act:.0f}% "
                    f"idle {100 * int(it['idle']) / act:.0f}% (level-1 run)"
                )
            print(f"  {role} {key[1][15:-1]} ch{key[2]} ({len(busiest)} such cores, {pk} pkts): {util}")
            phases = (
                ["tx_send", "tx_completion", "tx_conn"]
                if role == "TX"
                else ["rx_decode_admit", "rx_forward", "rx_local", "rx_completion", "rx_ack"]
            )
            total = 0.0
            for ph in phases:
                n = int(r[ph + "_count"]) / pk
                if n == 0:
                    continue
                c = cost(runs[PHASE_RUN[ph]], key, ph)
                if c is None:
                    continue
                total += c * n
                extra = ""
                if ph == "tx_send":
                    spin = cost(runs["m_spin"], key, "tx_txq_spin")
                    extra = f"   (of which txq spin {spin:.0f})"
                print(f"      {ph:16s} {c:6.1f} cycles x {n:5.2f}/pkt = {c * n:6.1f}{extra}")
            vph = "tx_visit" if role == "TX" else "rx_visit"
            nv = int(r[vph + "_count"]) / pk
            vc = cost(runs["m_visit"], key, vph)
            visits = vc * nv
            poll = visits - total
            print(
                f"      {'visit polling':16s} {vc:6.1f} cycles/visit x {nv:5.2f} visits/pkt = {visits:6.1f} total,"
                f" {poll:6.1f} beyond the phases above"
            )
            print(
                f"      {'outside visits':16s} {iv[1] - visits:6.1f}  (loop bookkeeping"
                f"{', idle' if role == 'RX' else ''}; = interval - visits)"
            )
            print(f"      {'= interval':16s} {iv[1]:6.1f}")


if __name__ == "__main__":
    main()
