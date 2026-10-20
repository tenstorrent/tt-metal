#!/usr/bin/env python3
"""Sender channels on E1: what moving some of the sending work to the receiving ERISC is worth once receive work has
moved to a DRISC.

Today E0 owns all four VC0 sender channels and E1 does all receive work. Here E1 also owns `m` of the sender channels
(router channels first; the worker channel ch0 stays on E0) and sends their packets on its own TX queue (TXQ1).

Sending side, from the compiled router with the polling flags (RESULTS §17, fabric_exp/timeline.html), not the
catalog's older polling parameters:
  per loop iteration:  loop 2; grouped free-slot read (4 + 2 per channel owned, 7 for one channel); teardown
                       check 11 on the core owning the worker channel
  per channel visit:   9 with nothing in flight (packet check + sent-count compare), 20 with packets in flight
                       (packet check + completion-counter read)
  per packet sent:     can-send check 12; send 72, of which 18 is the pointer update and the read before it
                       (paid once per b_ptr packets); credit to the producer 35 x 0.74 (paid once per b_ack)
  With one busy channel this gives 182 cycles per packet against 186.2 measured.
E1 as a sender: same costs. Its per-packet pointer update must be an L1 counter write, not a stream-register write
(only one RX queue takes register writes, RESULTS §16); assumed to cost the same. The receiving E1 then watches a
second ring whose arrival count is an L1 counter: +10 per iteration (estimated: L1 read 8 + branch).

Receiving side: fabric_model.evaluate for each receive placement over E1 and one DRISC (catalog costs, measured
DRISC handoffs), with the given batching. A core that both sends and receives runs one loop: per interval it does
max(1/k for receiving, sends/(busy channels x k) for sending) iterations and pays every per-iteration cost in each.

Traffic on one link, 4 KiB, per interval:
  both ways: each chip's sending side sends one packet and its receiving side receives one
  one way:   the sending chip's E0 and E1 share the sending; the receiving chip's E1 receives; nothing comes back
Channel load: `4 busy` = all four sender channels carry equal traffic (a loaded mesh); `1 busy` = one worker flow,
the other three channels idle (the benchmark case; moving them only moves their polling).

Not modelled: slot depth and the credit loop (splitting the receive ring in two halves each ring's slots unless
memory is added; README "Slot depth gates everything"), and ordering across the two TX queues (packets of one
sender channel always use one queue, so per-channel order holds).
"""
import argparse
import os
from dataclasses import replace

import fabric_model as fm

HERE = os.path.dirname(os.path.abspath(__file__))

MIXES = [
    ("local write E/W", "2d_single_ew"),
    ("local write N/S", "2d_single_ns"),
    ("scatter E/W", "2d_single_ew_scatter"),
    ("forward (3-hop middle)", "2d_3hop_middle"),
    ("multicast N/S (local + 2 fwd)", "2d_mcast_ns_router"),
]
HEADER = 96  # compiled 2D header on this mesh (HybridMeshPacketHeaderT<36>)
CHANNELS = 4
SECOND_RING_POLL = 10  # estimated: L1 counter read 8 + branch


class TxCosts:
    def __init__(self, send=72.0, ptr_part=18.0):
        self.loop = 2.0
        self.teardown = 11.0
        self.visit_idle = 9.0
        self.visit_busy = 20.0
        self.can_send = 12.0
        self.send = send
        self.ptr_part = ptr_part
        self.credit = 35.0 * 0.74

    def grouped(self, n):
        return 0.0 if n == 0 else (7.0 if n == 1 else 4.0 + 2.0 * n)

    def per_packet(self, b_ptr, b_ack):
        return self.can_send + self.send - self.ptr_part + self.ptr_part / b_ptr + self.credit / b_ack

    def per_iteration(self, owned, busy, worker):
        return (
            self.grouped(owned)
            + self.visit_busy * busy
            + self.visit_idle * (owned - busy)
            + (self.teardown if worker else 0.0)
        )


def tx_load(tx, owned, busy, worker, sends, k, b_ptr, b_ack, own_loop=True):
    """(cycles per interval, iterations per interval) for a core sending `sends` packets per interval from `busy`
    channels that each drain up to k packets per visit."""
    if owned == 0:
        return 0.0, 0.0
    iters = sends / (busy * k) if busy else 0.0
    per_iter = tx.per_iteration(owned, busy, worker) + (tx.loop if own_loop else 0.0)
    return sends * tx.per_packet(b_ptr, b_ack) + per_iter * iters, iters


def rx_parts(cat, scen, pl, cores, opts, k):
    """E1's receive work per interval, its loop + polls per iteration, the DRISC load and the NoC floor."""
    r = fm.evaluate(cat, scen, pl, cores, opts)
    if not r.valid:
        return None
    bd = r.breakdown.get("E1", {})
    work = sum(v for lbl, v in bd.items() if not lbl.startswith("loop + polls"))
    per_iter = sum(v for lbl, v in bd.items() if lbl.startswith("loop + polls")) * k  # evaluate ran 1/k iterations
    return work, per_iter, r.loads.get("D0", 0.0), r.floors["noc port"]


def split(busy_total, m):
    """Busy channels on (E0, E1) when E1 owns m router channels: the highest-numbered channels move first."""
    busy = [c < busy_total for c in range(CHANNELS)]  # ch0 (worker) is busy first
    e1 = list(range(CHANNELS - m, CHANNELS))
    e0 = [c for c in range(CHANNELS) if c not in e1]
    return (len(e0), sum(busy[c] for c in e0)), (len(e1), sum(busy[c] for c in e1))


def evaluate_case(cat, tx, scen, pl, cores, traffic, busy_total, m, k, b_ptr, b_ack, knobs):
    k_tx = knobs.k_tx or k  # packets a sending core takes from one channel per visit
    k_rx = knobs.k_rx or k  # packets a receiving core handles per loop iteration
    opts = fm.Options(payload=4096, k_drain=k_rx, b_ack=b_ack, b_ptr=b_ptr)
    parts = rx_parts(cat, scen, pl, cores, opts, k_rx)
    if parts is None:
        return None
    rx_work, rx_iter, drisc, noc_floor = parts
    rx_iter = max(rx_iter - knobs.e1_overhead_cut, 0.0)
    (e0_own, e0_busy), (e1_own, e1_busy) = split(busy_total, m)
    busy = e0_busy + e1_busy
    e0_share = e0_busy / busy
    e1_share = e1_busy / busy
    ring2 = (SECOND_RING_POLL if knobs.e1_tx_check is None else 0.0) if m > 0 else 0.0
    loads = {}
    # Sending side on E0: always its own core with its own loop.
    loads["E0"], _ = tx_load(tx, e0_own, e0_busy, True, e0_share, k_tx, b_ptr, b_ack)
    if e0_busy == 0 and e0_own:  # E0 owns only idle channels: it still visits them, at the pace of the link
        loads["E0"] = tx.per_iteration(e0_own, 0, True) + tx.loop
    # E1's sending work, without loop overhead (E1's loop is counted with its receive polls).
    e1_tx_iter = tx.per_iteration(e1_own, e1_busy, False) if e1_own else 0.0
    if e1_own and knobs.e1_tx_check is not None:  # E1's sender checks (and the second ring) at a fixed cost
        e1_tx_iter = knobs.e1_tx_check
    e1_tx_pkt = e1_share * tx.per_packet(b_ptr, b_ack)
    e1_tx_iters = e1_share / (e1_busy * k_tx) if e1_busy else 0.0
    if traffic == "both":
        iters = max(1.0 / k_rx, e1_tx_iters)
        loads["E1"] = rx_work + e1_tx_pkt + (rx_iter + ring2 + e1_tx_iter) * iters
    else:
        # The sending chip's E1 sends and still polls its (empty) receive rings every iteration. The receiving
        # chip's E1 receives and still visits its own (idle) sender channels every iteration.
        if e1_busy:
            loads["E1 send"] = e1_tx_pkt + (rx_iter + ring2 + e1_tx_iter) * e1_tx_iters
        idle_tx_iter = tx.per_iteration(e1_own, 0, False) if e1_own else 0.0
        if e1_own and knobs.e1_tx_check is not None:
            idle_tx_iter = knobs.e1_tx_check
        loads["E1 recv"] = rx_work + (rx_iter + ring2 + idle_tx_iter) / k_rx
    if drisc:
        loads["D0"] = drisc
    data = fm.data_wire_cycles(cat, 4096, HEADER) + fm.small_frame_cycles(cat) / b_ptr
    wire = data + (fm.credit_frame_cycles(cat) / b_ack if traffic == "both" else 0.0)
    floors = {"wire": wire, "noc port": noc_floor}
    interval, limit = max(((v, n) for n, v in list(loads.items()) + list(floors.items())), key=lambda x: x[0])
    return interval, limit, loads


def best(cat, tx, scen, cores, traffic, busy_total, m, k, b_ptr, b_ack, use_drisc, knobs):
    out = None
    for pl in fm.placements(cat, scen, cores):
        if pl["tx_completion"] != "E0":
            continue
        if not use_drisc and "D0" in pl.values():
            continue
        r = evaluate_case(cat, tx, scen, pl, cores, traffic, busy_total, m, k, b_ptr, b_ack, knobs)
        if r is None:
            continue
        key = (r[0], sorted(r[2].values(), reverse=True))
        if out is None or key < out[0]:
            out = (key, r, pl)
    return out[1], out[2]


ABBREV = {"E0": "E0", "E1": "E1", "E1 send": "E1s", "E1 recv": "E1r", "D0": "D", "wire": "w", "noc port": "noc"}


def table(cat, tx, traffic, busy_total, k, b_ptr, b_ack, knobs, detail=False):
    cores = fm.make_cores(1)
    ms = [0, 1, 2]
    print(
        f"\n### {'both ways' if traffic == 'both' else 'one way'}, {busy_total} busy sender channel"
        f"{'s' if busy_total > 1 else ''}, batching k_tx={knobs.k_tx or k} k_rx={knobs.k_rx or k} "
        f"b_ptr={b_ptr} b_ack={b_ack}"
    )
    print("Cycles per packet (limit). Columns: sender channels on E1 (m).")
    hdr = (
        "| receive work | "
        + " | ".join(f"E1 only, m={m}" for m in ms)
        + " | "
        + " | ".join(f"+DRISC, m={m}" for m in ms)
        + " |"
    )
    print(hdr)
    print("|" + " --- |" + " ---: |" * (2 * len(ms)))
    details = []
    for label, scen in MIXES:
        cells = []
        for use_drisc in (False, True):
            for m in ms:
                (interval, limit, loads), pl = best(
                    cat, tx, scen, cores, traffic, busy_total, m, k, b_ptr, b_ack, use_drisc, knobs
                )
                cells.append(f"{interval:.0f} ({ABBREV[limit]})")
                if detail:
                    lds = " ".join(f"{n}={v:.0f}" for n, v in loads.items())
                    rx_on_d = "+".join(p for p, c in pl.items() if c == "D0") or "-"
                    details.append(f"  {label:30s} {'DRISC' if use_drisc else 'E1   '} m={m}: {lds}   DRISC: {rx_on_d}")
        print(f"| {label} | " + " | ".join(cells) + " |")
    for d in details:
        print(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--send", type=float, default=72.0, help="send cost per packet (72: polling-flags build)")
    ap.add_argument("--detail", action="store_true")
    ap.add_argument(
        "--e1-overhead-cut",
        type=float,
        default=0.0,
        help="cycles taken off E1's receive loop + polls per iteration (today ~50-62, not yet broken down)",
    )
    ap.add_argument(
        "--e1-tx-check",
        type=float,
        default=None,
        help="fixed cost per iteration of E1's sender-channel checks and second ring, replacing the "
        "per-channel visits (e.g. a few cycles if grouped with E1's own ring read)",
    )
    ap.add_argument("--only", choices=["both", "one"], default=None)
    ap.add_argument("--k-tx", type=int, default=None, help="override packets per sender-channel visit")
    ap.add_argument("--k-rx", type=int, default=None, help="override packets the receiving core handles per iteration")
    ap.add_argument("--b-ack", type=int, default=None, help="override packets per completion ack")
    ap.add_argument("--b-ptr", type=int, default=None, help="override packets per pointer update")
    ap.add_argument("--k", type=int, default=None, help="only this batching level (1 or 4)")
    args = ap.parse_args()
    cat = fm.load_catalog()
    tx = TxCosts(send=args.send)
    wire = fm.data_wire_cycles(cat, 4096, HEADER) + fm.small_frame_cycles(cat)
    print(
        f"Wire, 4 KiB + {HEADER} B header: {wire:.1f} one way, {wire + fm.credit_frame_cycles(cat):.1f} with acks "
        f"for the other direction (b_ptr = b_ack = 1). Send cost {args.send:.0f}."
    )
    print(
        "Limits: E0, E1 (both ways), E1s / E1r (one way: sending chip's E1 / receiving chip's E1), D = DRISC, "
        "w = wire, noc = NoC port."
    )
    if args.e1_overhead_cut or args.e1_tx_check is not None:
        print(
            f"Sensitivity: E1 loop + polls cut by {args.e1_overhead_cut:.0f}; E1 sender checks per iteration "
            f"{'per channel' if args.e1_tx_check is None else f'{args.e1_tx_check:.0f}'}."
        )
    for traffic in (args.only,) if args.only else ("both", "one"):
        for busy_total in (4, 1):
            for k, b in ((args.k, args.k),) if args.k else ((1, 1), (4, 4)):
                table(cat, tx, traffic, busy_total, k, args.b_ptr or b, args.b_ack or b, args, args.detail)


if __name__ == "__main__":
    main()
