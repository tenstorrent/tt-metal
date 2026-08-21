#!/usr/bin/env python3
"""
DRAM read-consistency checker.

Detects the fault class where a GDDR channel trains OK, passes BIST, answers at normal
latency, and returns *corrupted data* on a subset of the bus -- e.g. one byte lane of a
32-bit interface silently returning stale bytes.

METHOD
  Each DRAM channel ("bank") is reachable through several aliased NOC endpoints. Reading
  the same address through two of them is a free, non-destructive read-integrity oracle:
  the ports front the same physical memory, so any disagreement is a proven read error --
  no ground truth needed, no writes, no core halts.

  Per endpoint we also re-read the same address N times to catch time non-determinism.

READ-ONLY. Safe against a live hang or alongside a running workload.

WHAT A PASS DOES NOT PROVE
  This compares read against read: it measures SELF-CONSISTENCY, NOT CORRECTNESS. Two
  reads can be identically wrong. Use it to decide *whether and where*, never *how bad*.
  See RULEBOOK.md.
"""

import argparse
import collections
import hashlib
import json
import sys
import time

# ---- predefined defaults (override with flags) ------------------------------------
ADDR = 0x10000000
BYTES = 128 * 1024
REPS = 3

# The top 16 MiB of every channel is register space in AXI mode (control regs, NOC_NIU,
# overlay streams from 0xFFB12000) and the top 2 KiB aliases MRISC instruction SRAM.
# Probing there yields phantom corruption.
MAX_ADDR = 0xFF000000

# Detection power saturates above 64 B per transaction; below that the fault is
# substantially under-reported (4 B reads showed ~19% of bytes wrong where 64 B showed 99%).
MIN_BYTES = 64

MAX_CHANNELS = 32
MAX_SUBCHANNELS = 8


def enumerate_banks(device):
    """Discover DRAM banks and their NOC ports from the device itself.

    ttexalens exposes DRAM as logical 'd<channel>,<subchannel>' coordinates; we walk them
    until translation fails. Nothing is hardcoded, so this follows harvesting and arch.
    """
    from ttexalens.coordinate import OnChipCoordinate

    banks = []
    for ch in range(MAX_CHANNELS):
        ports = []
        for sub in range(MAX_SUBCHANNELS):
            try:
                loc = OnChipCoordinate.create(f"d{ch},{sub}", device)
            except Exception:
                break
            ports.append((loc.to_str("noc0"), loc))
        if not ports:
            break
        banks.append(ports)
    return banks


def diff_positions(a, b):
    """Byte offsets where two equal-length buffers differ. Fast path for the common case."""
    if a == b:
        return []
    return [i for i in range(len(a)) if a[i] != b[i]]


def check_device(ctx, device, dev_id, addr, nbytes, reps):
    """Returns (bank_results, elapsed_seconds). One entry per bank, each with per-port rows."""
    from ttexalens import tt_exalens_lib as lib

    banks = enumerate_banks(device)
    results = []
    t0 = time.time()

    for bank_idx, ports in enumerate(banks):
        rows = []
        reads = {}  # port name -> list of `reps` buffers, or None on error

        for port_name, loc in ports:
            bufs = []
            err = None
            for _ in range(reps):
                try:
                    bufs.append(bytes(lib.read_from_device(loc, addr, device_id=dev_id, num_bytes=nbytes, context=ctx)))
                except Exception as e:
                    err = str(e)[:120]
                    break
            reads[port_name] = None if err else bufs
            if err:
                rows.append({"port": port_name, "error": err})

        # repeat: reads #2..#N vs #1 on the same port
        repeat_diff = {}
        for port_name, bufs in reads.items():
            if bufs is None:
                continue
            pos = set()
            for b in bufs[1:]:
                pos.update(diff_positions(bufs[0], b))
            repeat_diff[port_name] = pos

        # cross-port: each port's read #1 vs every OTHER port's read #1, same bank.
        # The ports alias one physical channel, so they must agree.
        xport_diff = {p: set() for p in reads if reads[p] is not None}
        example = {}
        names = [p for p in reads if reads[p] is not None]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                pos = diff_positions(reads[a][0], reads[b][0])
                if pos:
                    xport_diff[a].update(pos)
                    xport_diff[b].update(pos)
                    off = pos[0]
                    example.setdefault(a, (off, reads[a][0][off], b, reads[b][0][off]))
                    example.setdefault(b, (off, reads[b][0][off], a, reads[a][0][off]))

        for port_name, loc in ports:
            if reads[port_name] is None:
                continue
            rep = repeat_diff.get(port_name, set())
            xp = xport_diff.get(port_name, set())
            allpos = rep | xp
            row = {
                "port": port_name,
                "repeat_diffs": len(rep),
                "xport_diffs": len(xp),
                "mod4": dict(sorted(collections.Counter(o % 4 for o in allpos).items())),
                "sha256": [hashlib.sha256(b).hexdigest()[:16] for b in reads[port_name]],
            }
            if allpos and port_name not in example:
                off = min(allpos)
                row["first_bad"] = (off, reads[port_name][0][off], None, None)
            elif port_name in example:
                row["first_bad"] = example[port_name]
            rows.append(row)

        failed = any(r.get("repeat_diffs") or r.get("xport_diffs") for r in rows)
        errored = any("error" in r for r in rows)
        results.append({"bank": bank_idx, "rows": rows, "failed": failed, "errored": errored})

    return results, time.time() - t0


def main():
    ap = argparse.ArgumentParser(
        description="Read-only DRAM read-consistency check (cross-port + repeat-read oracle).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="A PASS proves self-consistency, NOT correctness -- two reads can be identically\n"
        "wrong. Never quote an error rate from this tool. See RULEBOOK.md.",
    )
    ap.add_argument("--devices", default="all", help="comma-separated device ids, or 'all' (default)")
    ap.add_argument("--addr", default=hex(ADDR), help=f"DRAM address to read (default {hex(ADDR)})")
    ap.add_argument("--bytes", type=int, default=BYTES, help=f"bytes per read (default {BYTES})")
    ap.add_argument("--reps", type=int, default=REPS, help=f"reads per endpoint (default {REPS})")
    ap.add_argument("--json", default=None, help="write full per-endpoint records here")
    args = ap.parse_args()

    addr = int(args.addr, 0)
    if addr >= MAX_ADDR:
        print(
            f"ERROR: --addr must be below {hex(MAX_ADDR)} -- the top 16 MiB of each channel is\n"
            f"       register space in AXI mode and the top 2 KiB aliases MRISC instruction SRAM.\n"
            f"       Probing there produces phantom corruption.",
            file=sys.stderr,
        )
        return 2
    if args.bytes < MIN_BYTES:
        print(
            f"ERROR: --bytes must be >= {MIN_BYTES}. The error rate is steeply read-size\n"
            f"       dependent and a smaller read will substantially under-report the fault.",
            file=sys.stderr,
        )
        return 2
    if args.reps < 2:
        print("ERROR: --reps must be >= 2.", file=sys.stderr)
        return 2

    try:
        from ttexalens.tt_exalens_init import init_ttexalens

        ctx = init_ttexalens()
    except Exception as e:
        print(f"ERROR: could not initialise ttexalens: {e}", file=sys.stderr)
        return 2

    by_id = {d.id: d for d in ctx.devices.values()}
    if args.devices == "all":
        dev_ids = sorted(by_id)
    else:
        try:
            dev_ids = [int(x, 0) for x in args.devices.split(",") if x.strip()]
        except ValueError:
            print(f"ERROR: bad --devices '{args.devices}'", file=sys.stderr)
            return 2
        missing = [d for d in dev_ids if d not in by_id]
        if missing:
            print(f"ERROR: device(s) {missing} not present; available {sorted(by_id)}", file=sys.stderr)
            return 2

    print(
        f"addr 0x{addr:08x}   {args.bytes // 1024} KB per read   {args.reps} reads per endpoint   "
        f"{len(dev_ids)} device(s)"
    )

    # Bank/port map, discovered from the device.
    probe_banks = enumerate_banks(by_id[dev_ids[0]])
    print(
        f"\ndiscovered {len(probe_banks)} banks x {len(probe_banks[0]) if probe_banks else 0} "
        f"NOC ports on device {dev_ids[0]}:"
    )
    for i, ports in enumerate(probe_banks):
        print(f"  bank {i}: {', '.join(p for p, _ in ports)}")

    all_results = {}
    print()
    for dev_id in dev_ids:
        res, elapsed = check_device(ctx, by_id[dev_id], dev_id, addr, args.bytes, args.reps)
        all_results[dev_id] = res
        n_ep = sum(len(b["rows"]) for b in res)
        print(f"  device {dev_id:3d}: {len(res)} banks, {n_ep} endpoints, {elapsed:6.1f}s")

    # ---- compact grid --------------------------------------------------------------
    nbanks = max(len(r) for r in all_results.values())
    print("\ndev    " + " ".join(f"b{i}" for i in range(nbanks)))
    for dev_id in dev_ids:
        cells = []
        for b in all_results[dev_id]:
            cells.append("⚠️" if b["errored"] else ("❌" if b["failed"] else "✅"))
        print(f"{dev_id:3d}    " + " ".join(cells))

    # ---- failure detail ------------------------------------------------------------
    bad_rows = []
    for dev_id in dev_ids:
        for b in all_results[dev_id]:
            for r in b["rows"]:
                if "error" in r or r.get("repeat_diffs") or r.get("xport_diffs"):
                    bad_rows.append((dev_id, b["bank"], r))

    if bad_rows:
        print("\ndev  bank  port   repeat-diffs  x-port-diffs   offsets mod 4        first bad")
        for dev_id, bank, r in bad_rows:
            if "error" in r:
                print(f"{dev_id:3d}  {bank:4d}  {r['port']:5s}   READ ERROR: {r['error']}")
                continue
            fb = r.get("first_bad")
            fb_s = ""
            if fb and fb[2] is not None:
                fb_s = f"+0x{fb[0]:x}  {fb[1]:02x} vs {fb[3]:02x} ({fb[2]})"
            elif fb:
                fb_s = f"+0x{fb[0]:x}  {fb[1]:02x}"
            print(
                f"{dev_id:3d}  {bank:4d}  {r['port']:5s}   {r['repeat_diffs']:12d}  "
                f"{r['xport_diffs']:12d}   {str(r['mod4']):18s}   {fb_s}"
            )

    n_bad_dev = len({d for d, _, _ in bad_rows})
    n_bad_bank = len({(d, b) for d, b, _ in bad_rows})
    n_ep_total = sum(len(b["rows"]) for r in all_results.values() for b in r)
    n_bank_total = sum(len(r) for r in all_results.values())
    print(
        f"\nsummary: {n_bad_dev} device(s), {n_bad_bank} bank(s), {len(bad_rows)} endpoint(s) FAILED; "
        f"{n_bank_total - n_bad_bank} of {n_bank_total} banks and "
        f"{n_ep_total - len(bad_rows)} of {n_ep_total} endpoints clean"
    )
    if not bad_rows:
        print(
            "NOTE: a clean result proves self-consistency, not correctness. Two reads can be\n"
            "      identically wrong. See RULEBOOK.md before concluding the DRAM is healthy."
        )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "addr": addr,
                    "bytes": args.bytes,
                    "reps": args.reps,
                    "results": {str(k): v for k, v in all_results.items()},
                },
                f,
                indent=1,
                default=str,
            )
        print(f"wrote {args.json}")

    return 1 if bad_rows else 0


if __name__ == "__main__":
    sys.exit(main())
