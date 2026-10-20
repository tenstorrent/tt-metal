#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Check that a cached active image's hot lines do not share L3 indices.

Under the 14-way LIM shape the L3 keeps two cache ways, and with the pipeline
input pinned in one of them every hart's cached data shares the other: one
128 KiB way, inclusive of the L2s and L1Ds (tt-isa-documentation
BlackholeA0/L2CPUTile/Caches.md § L3 Cache). This check takes an address's L3
index to be the address mod 128 KiB. Two hot lines with the same index evict
each other. For each image whose code is in the cached
local GDDR (ld/x280-gddr.ld) this lists, as index ranges, the loaded image
(.text through .data), the top 4 KiB of each hart's stack and the pinned input
(x280_pinned_input.h), and fails on any overlap between two of them. An image
linked into LIM is uncached and skipped.

Usage: check_l3_conflicts.py --readelf <riscv readelf> <image.elf | image dir>...
A directory stands for every .elf in it.
"""

import argparse
import pathlib
import re
import subprocess
import sys

WAY_BYTES = 128 * 1024
STACK_HOT_BYTES = 4 * 1024
HARTS = 4
# x280_pinned_input.h: X280_PINNED_INPUT_BASE, X280_PINNED_INPUT_SPAN.
PINNED_INPUT = (0x0A000000, 0x10C0)
# x280_address_map.h: X280_LOCAL_DDR_TILE_BASE; its cached alias spans 4 GiB.
CACHED_GDDR = (0x400030000000, 1 << 32)


def readelf(tool, *args):
    return subprocess.run([tool, *args], check=True, capture_output=True, text=True).stdout


def image_layout(tool, elf):
    """(start, end) of the loaded image, and the _stack_top and _stack_size symbols."""
    sections = {}
    for m in re.finditer(r"\]\s+(\S+)\s+\S+\s+([0-9a-f]+)\s+[0-9a-f]+\s+([0-9a-f]+)", readelf(tool, "-SW", elf)):
        sections[m.group(1)] = (int(m.group(2), 16), int(m.group(3), 16))
    symbols = {}
    for m in re.finditer(
        r"^\s*\d+:\s+([0-9a-f]+)\s+\d+\s+\S+\s+\S+\s+\S+\s+\S+\s+(\S+)$", readelf(tool, "-Ws", elf), re.M
    ):
        symbols[m.group(2)] = int(m.group(1), 16)
    missing = [n for n in (".text", ".data") if n not in sections] + [
        n for n in ("_stack_top", "_stack_size") if n not in symbols
    ]
    if missing:
        raise SystemExit(f"{elf}: no {', '.join(missing)}")
    text, data = sections[".text"], sections[".data"]
    return (text[0], data[0] + data[1]), symbols["_stack_top"], symbols["_stack_size"]


def index_ranges(start, size):
    """Half-open L3 index ranges [lo, hi) covering [start, start + size)."""
    if size >= WAY_BYTES:
        return [(0, WAY_BYTES)]
    lo = start % WAY_BYTES
    hi = lo + size
    return [(lo, hi)] if hi <= WAY_BYTES else [(lo, WAY_BYTES), (0, hi - WAY_BYTES)]


def conflicts(occupants):
    """Pairs of occupant names whose index ranges overlap."""
    found = []
    for i, (a, ra) in enumerate(occupants):
        for b, rb in occupants[i + 1 :]:
            if any(lo1 < hi2 and lo2 < hi1 for lo1, hi1 in ra for lo2, hi2 in rb):
                found.append((a, b))
    return found


def check(tool, elf):
    (start, end), stack_top, stack_size = image_layout(tool, elf)
    if not CACHED_GDDR[0] <= start < CACHED_GDDR[0] + CACHED_GDDR[1]:
        print(f"SKIP {elf}: .text at {start:#x} is not in the cached local GDDR")
        return True
    occupants = [("image", index_ranges(start, end - start)), ("pinned input", index_ranges(*PINNED_INPUT))]
    for hart in range(HARTS):
        top = stack_top - hart * stack_size
        occupants.append((f"hart {hart} stack top", index_ranges(top - STACK_HOT_BYTES, STACK_HOT_BYTES)))
    for name, ranges in occupants:
        print(f"  {name:20} " + " ".join(f"[{lo:#07x}, {hi:#07x})" for lo, hi in ranges))
    found = conflicts(occupants)
    for a, b in found:
        print(f"FAIL {elf}: {a} and {b} share L3 indices")
    if not found:
        print(f"PASS {elf}: image {end - start} bytes, no shared L3 indices")
    return not found


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--readelf", required=True)
    parser.add_argument("paths", nargs="+", type=pathlib.Path)
    args = parser.parse_args()
    elfs = [e for p in args.paths for e in (sorted(p.glob("*.elf")) if p.is_dir() else [p])]
    if not elfs:
        raise SystemExit(f"no .elf under {' '.join(map(str, args.paths))}")
    results = [check(args.readelf, str(elf)) for elf in elfs]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
