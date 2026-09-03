# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Read per-layer L1 dumps and answer two questions about #54876.

`PREFILL_DUMP_L1_PER_LAYER=1` makes the Kimi-K3 stack call `DumpDeviceMemoryState` before every
layer, which writes `detailed_memory_usage.csv` (every block's address, size and allocation status)
plus the usage summaries. This reads that pile and reports:

1. **Does L1 grow layer over layer?** A collision is consistent with something outliving its scope,
   and a per-layer trace is the only way to see it — a single-block probe cannot, because a leak
   that accumulates over layers looks like steady state inside one block.

2. **What sits at the address the clash names?** The failure only ever reports an offset ("L1 buffer
   allocated at 1563072 and static circular buffer region ends at 1563264"). The block list turns
   that offset into a size and a status, which is what distinguishes a tenant that could move to
   DRAM from one that must stay.

    python -m models.demos.deepseek_v3_d_p.scripts.analyze_l1_dumps generated --address 1563072
"""

import argparse
import csv
import re
from pathlib import Path


def _layer_of(path: Path):
    """Layer index from the `k3_L<n>_` prefix the dump was written under, or None."""
    m = re.search(r"k3_L(\d+)_", str(path))
    return int(m.group(1)) if m else None


def _blocks(path: Path):
    """(address, size, allocated) per block, tolerating column-name drift across versions."""
    out = []
    with open(path, newline="") as handle:
        for row in csv.reader(handle):
            cells = [c.strip() for c in row if c.strip()]
            nums = [c for c in cells if re.fullmatch(r"\d+", c)]
            if len(nums) >= 2:
                out.append((int(nums[0]), int(nums[1]), "1" if any("alloc" in c.lower() for c in cells) else "?"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="directory the reports were written under (usually generated/)")
    ap.add_argument("--address", type=int, default=1563072, help="the address the clash names")
    args = ap.parse_args()

    reports = sorted(Path(args.root).rglob("detailed_memory_usage.csv"))
    if not reports:
        raise SystemExit(f"no detailed_memory_usage.csv under {args.root}")
    print(f"{len(reports)} report(s)\n")

    print("layer   blocks    total bytes   max address")
    trace = []
    for path in reports:
        blocks = _blocks(path)
        if not blocks:
            continue
        layer = _layer_of(path)
        total = sum(size for _, size, _ in blocks)
        top = max(addr for addr, _, _ in blocks)
        trace.append((layer, total))
        print(f"{str(layer):>5}   {len(blocks):>6}   {total:>12,}   {top:>11,}")

    known = [(l, t) for l, t in trace if l is not None]
    if len(known) >= 2:
        first, last = known[0], known[-1]
        growth = last[1] - first[1]
        print(f"\nlayer {first[0]} -> {last[0]}: {growth:+,} bytes")
        print(
            "  GROWS -> something outlives its scope; find what is allocated per layer and not freed"
            if growth > 0
            else "  FLAT -> no leak across layers; the collision is placement, not lifetime"
        )

    print(f"\nblocks containing address {args.address:,}:")
    hit = False
    for path in reports:
        for addr, size, status in _blocks(path):
            if addr <= args.address < addr + size:
                print(f"  {path}: addr {addr:,} size {size:,} ({status}) — offset {args.address - addr}")
                hit = True
                break
    if not hit:
        print("  none — the address is not inside any reported block, so the tenant is not an")
        print("  L1 *tensor*. That points at a static CB reservation rather than a buffer.")


if __name__ == "__main__":
    main()
