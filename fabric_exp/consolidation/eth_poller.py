#!/usr/bin/env python3
"""eth_poller.py OUT_PREFIX --start-log LOG [--start-pattern P] [--stop-pattern P] [--max-s S] [--wait-s S]
[--chips 0,1]: polls the Ethernet link state and error counters of the fabric router cores while a test runs.

Targets: every connected Ethernet core whose launch message holds a router image (pc_sampler.py --only-loaded's rule),
read round-robin from the test's START_PATTERN until STOP_PATTERN or --max-s. Per core and sweep:
  l1    0x7C400..0x7D040 (3136 B, one read): the base firmware's debug buffer (link down / recovery / runtime
        retrain and reinit counts, last-check timestamps, API metrics, metal's PTP tenure trace) and boot results
        (eth_status: port and training state, heartbeat[0..3]; eth_live_status: retrain_count, rx_link_up,
        link_flap_count, link_poll_alive_count, frames / bytes TX and RX with bad-FCS and dropped, corrected and
        uncorrected FEC codewords, TXQ resend and RXQ drop totals; these are a snapshot base FW refreshes from
        ERISC0's link status check, at most once per second) and the base FW mailboxes;
  mbx   0x590..0x5D0 (64 B): the metal mailbox's go message and link_status_check_timestamp (next check due);
  regs  PCS_STATUS (0xFFB9800C, live link up = 1), ERR_STAT (0xFFB980D8, latched, reading does not clear it), and
        per TX queue 0..2 CMD, STATUS, TRANSFER_CNT, PKT_START_CNT, PKT_END_CNT, WORD_CNT, RESEND_CNT.
Once at the start and once at the end: every documented TXQ register (hang_capture.py's TXQ_OFFSETS).

Read-only: tt-exalens NoC reads in safe mode (eth_mac_pcs_regs is unsafe and never touched), no debug-bus selects,
no writes. Not read: the PTP timer and the RISC wall clock (reading their low word latches the high word the
firmware reads next). The device is opened as pc_sampler.py opens it, only after the test logged START_PATTERN.
Output: OUT_PREFIX.npz (records: t, target, regs[23], mbx[64 B], l1[3136 B]; txq_start / txq_end) and
OUT_PREFIX.json (targets, layout, rates, errors)."""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hang_capture import TXQ_BASE, TXQ_OFFSETS, TXQ_STRIDE, NUM_TXQ  # noqa: E402
from pc_sampler import core_layout  # noqa: E402

L1_LO, L1_HI = 0x7C400, 0x7D040  # debug_buf_t (2 KiB) + boot_results_t (1 KiB) + base FW mailboxes (64 B)
MBX_LO, MBX_HI = 0x590, 0x5D0  # mailboxes_t (at 0x100) go_messages .. link_status_check_timestamp (+1208)
PCS_STATUS, ERR_STAT = 0xFFB9800C, 0xFFB980D8
TXQ_POLL = (0x04, 0x08, 0x30, 0x34, 0x3C, 0x40, 0x68)  # CMD, STATUS, TRANSFER_CNT, PKT_START, PKT_END, WORD, RESEND
REGS = [PCS_STATUS, ERR_STAT] + [TXQ_BASE + q * TXQ_STRIDE + o for q in range(NUM_TXQ) for o in TXQ_POLL]
REG_NAMES = ["PCS_STATUS", "ERR_STAT"] + [f"txq{q}_{TXQ_OFFSETS[o]}" for q in range(NUM_TXQ) for o in TXQ_POLL]
REC = np.dtype([("t", "<f8"), ("i", "<u2"), ("regs", "<u4", (len(REGS),)), ("mbx", "u1", (MBX_HI - MBX_LO,)),
                ("l1", "u1", (L1_HI - L1_LO,))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--start-log", required=True)
    ap.add_argument("--start-pattern", default="Running Test:")
    ap.add_argument("--stop-pattern", default="Tests: ")
    ap.add_argument("--max-s", type=float, default=30.0)
    ap.add_argument("--wait-s", type=float, default=300.0)
    ap.add_argument("--chips", default="")
    args = ap.parse_args()

    def log_has(pat):
        try:
            with open(args.start_log, errors="replace") as f:
                return pat in f.read()
        except FileNotFoundError:
            return False

    t0 = time.time()
    while not log_has(args.start_pattern):
        if time.time() - t0 > args.wait_s:
            print("poller: start pattern never appeared", flush=True)
            return 2
        time.sleep(0.05)
    print(f"poller: start seen after {time.time() - t0:.1f} s; opening the device", flush=True)
    os.environ.pop("TT_VISIBLE_DEVICES", None)
    from ttexalens.tt_exalens_init import init_ttexalens
    from ttexalens.context import NocId

    ctx = init_ttexalens(noc_id=NocId.NOC0, safe_mode=True)
    conns = ctx.cluster_descriptor.get_ethernet_connections()
    want = {int(c) for c in args.chips.split(",")} if args.chips else None
    targets, layouts, errors = [], {}, {}
    for chip_id, dev in sorted(ctx.devices.items()):
        if want is not None and chip_id not in want:
            continue
        active = set(conns.get(chip_id, {}).keys())
        for loc in dev.get_block_locations("eth"):
            chan = int(loc.to("logical")[0][1])
            if chan not in active:
                continue
            lay = core_layout(loc)
            if not lay["erisc"][0]["text_size"]:
                continue
            layouts[f"{chip_id}/{chan}"] = lay
            targets.append((chip_id, chan, loc))
    print(f"poller: {len(targets)} router cores", flush=True)

    def txq_all():
        out = {}
        for chip_id, chan, loc in targets:
            d = {}
            for q in range(NUM_TXQ):
                for o, name in TXQ_OFFSETS.items():
                    try:
                        d[f"txq{q}_{name}"] = loc.noc_read32(TXQ_BASE + q * TXQ_STRIDE + o)
                    except Exception as ex:  # noqa: BLE001
                        errors[f"{chip_id}/{chan} txq{q} {name}: {type(ex).__name__}"] = 1
            out[f"{chip_id}/{chan}"] = d
        return out

    txq_start = txq_all()
    recs = []
    t_start = time.time()
    stop_check, sweeps = t_start, 0
    while True:
        for i, (chip_id, chan, loc) in enumerate(targets):
            try:
                t = time.time() - t_start
                regs = [loc.noc_read32(a) for a in REGS]
                mb = bytearray(MBX_HI - MBX_LO)
                loc.noc_read(MBX_LO, mb)
                l1 = bytearray(L1_HI - L1_LO)
                loc.noc_read(L1_LO, l1)
                recs.append((t, i, regs, np.frombuffer(bytes(mb), np.uint8), np.frombuffer(bytes(l1), np.uint8)))
            except Exception as ex:  # noqa: BLE001
                k = f"{chip_id}/{chan}: {type(ex).__name__}: {str(ex)[:120]}"
                errors[k] = errors.get(k, 0) + 1
        sweeps += 1
        now = time.time()
        if now - t_start > args.max_s:
            print("poller: max time reached", flush=True)
            break
        if now - stop_check > 0.25:
            stop_check = now
            if log_has(args.stop_pattern):
                print("poller: stop pattern seen", flush=True)
                break
    dur = time.time() - t_start
    txq_end = txq_all()
    arr = np.zeros(len(recs), dtype=REC)
    for k, r in enumerate(recs):
        arr[k] = r
    np.savez_compressed(args.out + ".npz", recs=arr)
    meta = {
        "targets": [{"chip": c, "chan": ch} for c, ch, _ in targets],
        "layouts": layouts,
        "reg_names": REG_NAMES,
        "l1_lo": L1_LO, "mbx_lo": MBX_LO,
        "txq_start": txq_start, "txq_end": txq_end,
        "t_start_epoch": t_start,
        "duration_s": round(dur, 3),
        "sweeps": sweeps,
        "sweep_ms": round(1e3 * dur / max(1, sweeps), 2),
        "records": len(recs),
        "errors": errors,
    }
    with open(args.out + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    print(f"poller: {len(recs)} records, {sweeps} sweeps in {dur:.2f} s ({meta['sweep_ms']} ms/sweep), "
          f"{sum(errors.values())} read errors", flush=True)
    try:
        from ttexalens.tt_exalens_init import cleanup_global_context
        cleanup_global_context()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
