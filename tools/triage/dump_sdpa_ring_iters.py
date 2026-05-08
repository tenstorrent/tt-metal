#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Dump (ring_iter, ring_id) pairs from the watcher ring buffer for every tensix
worker core on every device, over PCIe via tt-exalens. No smart parsing —
just reads the raw data[] array and prints pairs.

The kernel writes:
    data[0]=ring_iter_0, data[1]=ring_id_0, data[2]=ring_iter_1, ...

Usage:
    python tools/triage/dump_sdpa_ring_iters.py
    python tools/triage/dump_sdpa_ring_iters.py --devices 0,1,2
    python tools/triage/dump_sdpa_ring_iters.py --elf /path/to/brisc.elf
    python tools/triage/dump_sdpa_ring_iters.py --csv > rings.csv
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

from ttexalens.tt_exalens_init import init_ttexalens
from ttexalens.tt_exalens_lib import read_from_device
from ttexalens.server import FileAccessApi
from ttexalens.elf.parsed import read_elf
from ttexalens.memory_access import FixedMemoryAccess


def find_brisc_elf(repo_root: str) -> str:
    candidates: list[str] = []
    candidates += glob.glob(os.path.join(repo_root, "tt_metal/pre-compiled/*/brisc/brisc.elf"))
    candidates += glob.glob(os.path.join(repo_root, "built/*/firmware/brisc/brisc.elf"))
    candidates += glob.glob(os.path.join(repo_root, "built/firmware/brisc/brisc.elf"))
    if not candidates:
        raise FileNotFoundError("Could not auto-locate a brisc.elf. Pass --elf explicitly.")
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def resolve_data_addr(elf_path: str) -> tuple[int, int]:
    """Returns (data_addr, num_elements) for mailboxes.watcher.debug_ring_buf.data."""
    parsed = read_elf(FileAccessApi(), elf_path)
    fake_mem = FixedMemoryAccess(b"\x00" * 4096)
    mailboxes = parsed.get_global("mailboxes", fake_mem)
    data = mailboxes.watcher.debug_ring_buf.data
    elements = parsed.get_constant("DEBUG_RING_BUFFER_ELEMENTS")
    return data.get_address(), int(elements)


def main() -> int:
    ap = argparse.ArgumentParser(description="Dump SDPA ring_iter/ring_id pairs from watcher ring buffer (PCIe)")
    ap.add_argument("--elf", help="Path to any brisc.elf (auto-detected if omitted)")
    ap.add_argument("--devices", help="Comma-separated device ids (default: all visible)")
    ap.add_argument("--csv", action="store_true", help="Emit CSV: device_id,core,pair_index,ring_iter,ring_id")
    args = ap.parse_args()

    repo_root = os.environ.get(
        "TT_METAL_HOME", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    elf_path = args.elf or find_brisc_elf(repo_root)
    data_addr, num_elements = resolve_data_addr(elf_path)
    num_bytes = num_elements * 4
    print(f"# elf={elf_path}", file=sys.stderr)
    print(f"# data[] @ 0x{data_addr:x}, {num_elements} uint32 ({num_bytes} bytes)", file=sys.stderr)

    context = init_ttexalens()
    if args.devices:
        device_ids = [int(x) for x in args.devices.split(",") if x.strip()]
    else:
        device_ids = sorted(int(d) for d in context.devices.keys())
    print(f"# devices: {device_ids}", file=sys.stderr)

    if args.csv:
        print("device_id,core,pair_index,ring_iter,ring_id")

    for did in device_ids:
        device = context.devices[did]
        cores = device.get_block_locations(block_type="functional_workers")
        for core in cores:
            buf = read_from_device(core, data_addr, device_id=did, num_bytes=num_bytes, context=context)
            words = [int.from_bytes(buf[i * 4 : (i + 1) * 4], "little") for i in range(num_elements)]
            core_str = str(core)
            if args.csv:
                for i in range(0, num_elements, 2):
                    ri = words[i]
                    rid = words[i + 1] if i + 1 < num_elements else 0
                    print(f"{did},{core_str},{i // 2},0x{ri:08x},0x{rid:08x}")
            else:
                pairs = " ".join(
                    f"(0x{words[i]:08x},0x{words[i+1]:08x})" if i + 1 < num_elements else f"(0x{words[i]:08x},)"
                    for i in range(0, num_elements, 2)
                )
                print(f"dev={did} core={core_str} {pairs}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
