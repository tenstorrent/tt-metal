#!/usr/bin/env python3
"""Per-packet TX/RX cycle budget for the neighbor-exchange (speedy path) size sweep.

Each phase's per-occurrence cost comes from the run that timed only that phase, minus the timer's calibrated
in-phase cycles; occurrence counts per packet come from the counters-only run; the total is the profiler-off
interval. Expects runs nbr_notel, nb_cnt, nb_iter, nb_send, nb_spin, nb_compl, nb_rx, nb_visit, nb_tx30, nb_rx30,
nb_skip under fabric_exp/results.
"""
import csv
import re

RES = "fabric_exp/results"
PHASE_RUN = {
    "tx_send": "nb_send",
    "tx_txq_spin": "nb_spin",
    "tx_completion": "nb_compl",
    "rx_decode_admit": "nb_rx",
    "rx_local": "nb_rx",
    "rx_completion": "nb_rx",
    "tx_visit": "nb_visit",
    "rx_visit": "nb_visit",
}


def size_of(test):
    return int(re.search(r"size_(\d+)", test).group(1))


def load_phase(label):
    """Busiest data-carrying core per (size, erisc)."""
    best = {}
    for r in csv.DictReader(open(f"{RES}/{label}_phase.csv")):
        key = (size_of(r["test"]), r["erisc"])
        if key not in best or int(r["pkts"]) > int(best[key]["pkts"]):
            best[key] = r
    return best


def load_interval(label):
    return {
        int(r["packet_size"]): 1.35e9 / float(r["avg_packets_per_s"])
        for r in csv.DictReader(open(f"{RES}/{label}.csv"))
    }


def per_occurrence(r, phase):
    n = int(r[phase + "_count"])
    if n == 0:
        return None
    return int(r[phase + "_ticks"]) / n - (int(r["timer_cal"]) & 0xFFFFFFFF) / 100.0


def main():
    runs = {label: load_phase(label) for label in set(PHASE_RUN.values()) | {"nb_cnt", "nb_iter"}}
    base, tx30, rx30, skip = (load_interval(x) for x in ("nbr_notel", "nb_tx30", "nb_rx30", "nb_skip"))
    for size in sorted(base):
        iv = base[size]
        print(
            f"\n=== {size} B: {iv:.1f} cycles/packet  (TX+30 -> {tx30[size] - iv:+.0f}, RX+30 -> {rx30[size] - iv:+.0f},"
            f" skip local delivery -> {skip[size] - iv:+.0f})"
        )
        for erisc, role, phases in (
            ("0", "TX", ["tx_send", "tx_completion"]),
            ("1", "RX", ["rx_decode_admit", "rx_local", "rx_completion"]),
        ):
            cnt = runs["nb_cnt"][(size, erisc)]
            pk = int(cnt["pkts"])
            it = runs["nb_iter"][(size, erisc)]
            act = int(it["busy"]) + int(it["blocked"]) + int(it["idle"])
            print(
                f"  {role}: level-1 busy {100 * int(it['busy']) / act:.0f}% blocked {100 * int(it['blocked']) / act:.0f}%"
                f" idle {100 * int(it['idle']) / act:.0f}%"
            )
            total = 0.0
            for ph in phases:
                c = per_occurrence(runs[PHASE_RUN[ph]][(size, erisc)], ph)
                n = int(cnt[ph + "_count"]) / pk
                if c is None or n == 0:
                    continue
                total += c * n
                extra = ""
                if ph == "tx_send":
                    extra = f"  (txq spin inside: {per_occurrence(runs['nb_spin'][(size, erisc)], 'tx_txq_spin'):.1f})"
                print(f"    {ph:16s} {c:6.1f} x {n:5.2f}/pkt = {c * n:6.1f}{extra}")
            vph = "tx_visit" if role == "TX" else "rx_visit"
            vc = per_occurrence(runs["nb_visit"][(size, erisc)], vph)
            nv = int(cnt[vph + "_count"]) / pk
            print(
                f"    {'other visit work':16s} {vc * nv - total:6.1f}  (visit {vc:.1f} x {nv:.2f}/pkt minus the phases above)"
            )
            print(
                f"    {'outside visits':16s} {iv - vc * nv:6.1f}  (loop bookkeeping{' and idle' if role == 'RX' else ''})"
            )


if __name__ == "__main__":
    main()
