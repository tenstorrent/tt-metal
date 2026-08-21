#!/usr/bin/env python3
"""
REPRO. Standalone L1 pattern test for the dispatch cores' prefetch ring buffer.

No Metal, no build, no mesh, no model -- just ttexalens. Seconds per device.
If the fault is an L1 cell, this reproduces it directly and you never run the model again.

The prefetch ring buffer is 1024 KB (864 KB with >1 HW CQ) rooted at scratch_db_base_
(dispatch_settings.cpp:69). A stuck bit anywhere in it silently corrupts one byte of
whatever program binary lands there, on every launch, forever.

WRITES to L1. Requires the device to be free -- kill any Metal process first.
Contents are saved and restored per block; a block that fails to restore is reported
(expected at a genuinely faulty location).

  --dry-run   read only, no writes: validates plumbing and prints the plan
"""
import argparse, collections, os, random, sys
from ttexalens.tt_exalens_init import init_ttexalens
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.memory_access import create_l1_memory_access

BLOCK = 4096


def device_busy():
    """True if any process holds a /dev/tenstorrent node."""
    holders = set()
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            fdd = f"/proc/{pid}/fd"
            for fd in os.listdir(fdd):
                if "tenstorrent" in os.readlink(os.path.join(fdd, fd)):
                    holders.add(pid)
                    break
        except (OSError, PermissionError):
            continue
    return holders


def patterns(nbytes, seed):
    rnd = random.Random(seed)
    yield "0x00", bytes(nbytes)
    yield "0xff", b"\xff" * nbytes
    yield "0x55", b"\x55" * nbytes
    yield "0xaa", b"\xaa" * nbytes
    yield "walk1", bytes((1 << (i % 8)) for i in range(nbytes))
    yield "walk0", bytes(0xFF ^ (1 << (i % 8)) for i in range(nbytes))
    yield "addr", bytes((i * 7 + 0x5A) & 0xFF for i in range(nbytes))
    yield "random", bytes(rnd.randrange(256) for _ in range(nbytes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", default="all")
    ap.add_argument("--cores", default="16-2,16-3", help="dispatch cores, noc0 (from triage dump_fast_dispatch)")
    ap.add_argument("--start", type=lambda s: int(s, 0), default=0x0F0000)
    ap.add_argument(
        "--len",
        type=lambda s: int(s, 0),
        default=0x020000,
        help="bytes to sweep (default 128 KB around the known site; "
        "use --len 0x100000 --start <scratch_db_base> for the whole ring buffer)",
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="write even if a process holds the device")
    a = ap.parse_args()

    busy = device_busy()
    if busy and not a.dry_run and not a.force:
        print(
            f"REFUSING: pid(s) {sorted(busy)} hold /dev/tenstorrent. Writing to dispatch-core L1\n"
            f"          under a live Metal process will corrupt it. Kill it, or pass --force.",
            file=sys.stderr,
        )
        return 2
    if busy:
        print(
            f"note: pid(s) {sorted(busy)} hold the device" f"{' (dry run, read-only)' if a.dry_run else ' -- FORCED'}\n"
        )

    ctx = init_ttexalens()
    by = {d.id: d for d in ctx.devices.values()}
    ids = sorted(by) if a.devices == "all" else [int(x, 0) for x in a.devices.split(",")]
    cores = [c.strip() for c in a.cores.split(",")]
    pats = list(patterns(BLOCK, a.seed))

    print(
        f"sweep L1 [0x{a.start:06x}, 0x{a.start + a.len:06x})  "
        f"{a.len // BLOCK} blocks x {BLOCK} B  x {len(pats)} patterns"
    )
    print(f"cores {cores}  devices {ids}" f"{'   [DRY RUN -- no writes]' if a.dry_run else ''}\n")

    total_bad = 0
    for dev_id in ids:
        for core in cores:
            try:
                mem = create_l1_memory_access(OnChipCoordinate.create(core, by[dev_id], "noc0"))
            except Exception as e:
                print(f"  device {dev_id:3d} {core}: no L1 here ({str(e)[:44]})")
                continue

            bad = collections.Counter()  # (addr, bit) -> failing patterns
            restore_fail = []
            for off in range(a.start, a.start + a.len, BLOCK):
                orig = bytearray(BLOCK)
                try:
                    mem.read(off, orig)
                except Exception as e:
                    print(f"  device {dev_id:3d} {core} @0x{off:06x}: read error {str(e)[:40]}")
                    break
                if a.dry_run:
                    continue
                for name, pat in pats:
                    mem.write(off, pat)
                    back = bytearray(BLOCK)
                    mem.read(off, back)
                    for i in range(BLOCK):
                        if back[i] != pat[i]:
                            x = back[i] ^ pat[i]
                            for b in range(8):
                                if x >> b & 1:
                                    bad[(off + i, b)] += 1
                                    if bad[(off + i, b)] == 1:
                                        print(
                                            f"  device {dev_id:3d} {core}: L1 0x{off + i:06x} bit {b}"
                                            f"  wrote {pat[i]:02x} read {back[i]:02x}  [{name}]"
                                        )
                mem.write(off, bytes(orig))
                chk = bytearray(BLOCK)
                mem.read(off, chk)
                if bytes(chk) != bytes(orig):
                    restore_fail.append(off)

            if a.dry_run:
                print(f"  device {dev_id:3d} {core}: readable, {a.len // BLOCK} blocks OK")
                continue
            if bad:
                total_bad += 1
                addrs = sorted({ad for ad, _ in bad})
                print(
                    f"  device {dev_id:3d} {core}: *** FAIL *** {len(addrs)} address(es), "
                    f"{len(bad)} (address,bit) pair(s)"
                )
                for ad, b in sorted(bad):
                    print(f"       0x{ad:06x} bit {b}: failed {bad[(ad, b)]} of {len(pats)} patterns")
            else:
                print(f"  device {dev_id:3d} {core}: PASS")
            if restore_fail:
                print(
                    f"       restore failed on {len(restore_fail)} block(s): " f"{[hex(x) for x in restore_fail[:4]]}"
                )

    print(f"\n{total_bad} core(s) FAILED")
    return 1 if total_bad else 0


if __name__ == "__main__":
    sys.exit(main())
