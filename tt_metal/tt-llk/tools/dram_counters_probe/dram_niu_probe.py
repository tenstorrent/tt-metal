#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Do the DRAM tile NIU read counters move at the address the headers give? Read-only, any architecture.

For each DRAM tile the probe reads RD_DATA_WORD_SENT on both NoCs, reads SIZE bytes of DRAM from that tile over the
host path (a known amount of read traffic into the tile), reads the counter again and prints the change next to the
expected SIZE / word_bytes (plus a handful of words for the probe's own register reads).

  python dram_niu_probe.py --arch wormhole --mode umd
  python dram_niu_probe.py --arch wormhole --mode exalens
  python dram_niu_probe.py --arch blackhole --mode umd --size 65536

A counter that never changes, or an access error, means the address, instance or coordinate rule for that
architecture is wrong. Exit code 0 when at least one tile moved by about the expected amount.
"""

import argparse
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import _common as c


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arch", required=True, choices=("blackhole", "wormhole"))
    p.add_argument("--mode", default="umd", choices=("umd", "exalens"))
    p.add_argument("--size", type=int, default=65536)
    p.add_argument("--device", default=c.default_device_path())
    args = p.parse_args()
    if c.card_busy(args.device):
        c.fail(f"{args.device} is open by another process; not reading")
    layout = c.headers.dram_layout(args.arch)
    expected = args.size // layout.word_bytes
    reader = c.make_reader(args.mode, args.device)
    off = c.NIU_COUNTERS["RD_DATA_WORD_SENT"]
    good = 0
    print(
        f"arch {args.arch}, word {layout.word_bytes} B, reading {args.size} B per tile, expect about {expected} words"
    )
    for tile in reader.tiles():
        row = []
        for noc in (0, 1):
            addr = c.niu_addr(tile, noc, off, args.arch)
            try:
                before = reader.read32(tile, addr)
            except Exception as e:  # noqa: BLE001
                row.append(f"noc{noc} {hex(addr)} ERROR {e!r}"[:100])
                continue
            reader.read_bytes(tile, 0, args.size)
            after = reader.read32(tile, addr)
            delta = c.wrap_delta(after, before)
            row.append(f"noc{noc} {hex(addr)} {before}->{after} delta {delta}")
            if expected <= delta <= expected + 64:
                good += 1
        print(f"tile {tile}: " + " | ".join(row))
    print(f"{good} tile/NoC pairs moved by about the expected amount")
    raise SystemExit(0 if good else 1)


if __name__ == "__main__":
    main()
