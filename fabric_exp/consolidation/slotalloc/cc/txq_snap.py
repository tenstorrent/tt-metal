#!/usr/bin/env python3
"""txq_snap.py OUT.json --start-log LOG [--start-pattern P] [--stop-pattern P] [--period-s S] [--max-s S] [--chips ..]:
periodic read-only snapshots of the Ethernet TX queues' counters on every connected Ethernet core while a test runs
(credit-cadence A/B, TRID_ALLOCATION.md, 2026-11-04).

Per snapshot and core: TXQ0 and TXQ1 TRANSFER_CNT (0x30) and RESEND_CNT (0x68), at 0xFFB90000 + q * 0x1000
(tt_eth_ss_regs.h; hang_capture.py's TXQ_OFFSETS; RESULTS §16: TRANSFER_CNT is exact, PKT_* undercount at >= 4000 B).
In two-ERISC mode E0 sends the router's data on TXQ0 and E1 its credits on TXQ1.

The device is opened as pc_sampler.py opens it (tt-exalens, safe mode: no reset, no firmware load, no core init),
only after the test has logged START_PATTERN, so the test's own device init is never overlapped; the only device
accesses are NoC reads (4 per core and snapshot). Snapshots every PERIOD_S until STOP_PATTERN appears in the log or
MAX_S passes. Output: cores (chip, channel, connected peer) and snapshots (host time of each core's reads, the four
counters per core); cc_report.py picks the steady window."""
import argparse
import json
import os
import struct
import sys
import time

TXQ_BASE = 0xFFB90000
TXQ_STRIDE = 0x1000
REGS = {"transfer": 0x30, "resend": 0x68}
QUEUES = (0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--start-log", required=True)
    ap.add_argument("--start-pattern", default="Running Test:")
    ap.add_argument("--stop-pattern", default="Tests: ")
    ap.add_argument("--period-s", type=float, default=1.0)
    ap.add_argument("--max-s", type=float, default=90.0)
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
            print("txq_snap: start pattern never appeared", flush=True)
            return 2
        time.sleep(0.05)
    print(f"txq_snap: start seen after {time.time() - t0:.1f} s; opening the device", flush=True)

    os.environ.pop("TT_VISIBLE_DEVICES", None)
    from ttexalens.tt_exalens_init import init_ttexalens
    from ttexalens.context import NocId

    ctx = init_ttexalens(noc_id=NocId.NOC0, safe_mode=True)
    conns = ctx.cluster_descriptor.get_ethernet_connections()
    want = {int(c) for c in args.chips.split(",")} if args.chips else None
    cores = []  # (chip, chan, loc, peer)
    for chip_id, dev in sorted(ctx.devices.items()):
        if want is not None and chip_id not in want:
            continue
        active = conns.get(chip_id, {})
        for loc in dev.get_block_locations("eth"):
            chan = int(loc.to("logical")[0][1])
            if chan in active:
                cores.append((chip_id, chan, loc, list(active[chan])))
    print(f"txq_snap: {len(cores)} connected Ethernet cores", flush=True)

    buf = bytearray(4)
    snaps, errors = [], {}
    t_start = time.time()
    while True:
        snap = {"t": round(time.time() - t_start, 4), "cores": []}
        for chip_id, chan, loc, _ in cores:
            vals, t_core = {}, time.time() - t_start
            for q in QUEUES:
                for name, off in REGS.items():
                    try:
                        loc.noc_read(TXQ_BASE + q * TXQ_STRIDE + off, buf)
                        vals[f"q{q}_{name}"] = struct.unpack("<I", buf)[0]
                    except Exception as ex:  # noqa: BLE001 - recorded, the loop goes on
                        k = f"{chip_id}/{chan}: {type(ex).__name__}: {str(ex)[:100]}"
                        errors[k] = errors.get(k, 0) + 1
                        vals[f"q{q}_{name}"] = None
            snap["cores"].append({"t": round(t_core, 4), **vals})
        snaps.append(snap)
        if log_has(args.stop_pattern):
            print("txq_snap: stop pattern seen", flush=True)
            break
        if time.time() - t_start > args.max_s:
            print("txq_snap: max time reached", flush=True)
            break
        time.sleep(max(0.0, args.period_s - (time.time() - t_start - snap["t"])))
    meta = {
        "cores": [{"chip": c, "chan": ch, "peer": p} for c, ch, _, p in cores],
        "snapshots": snaps,
        "period_s": args.period_s,
        "errors": errors,
    }
    with open(args.out, "w") as f:
        json.dump(meta, f)
    print(f"txq_snap: {len(snaps)} snapshots of {len(cores)} cores, {sum(errors.values())} read errors", flush=True)
    try:
        from ttexalens.tt_exalens_init import cleanup_global_context
        cleanup_global_context()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
