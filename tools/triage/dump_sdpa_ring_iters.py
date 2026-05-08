#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Dump (ring_iter, ring_id) pairs that ring_joint_reader.cpp wrote into the
watcher ring buffer for every tensix worker core on every device.

Standalone — does NOT require the host process to still be running and does
NOT require inspector logs. Reads L1 over PCIe via tt-exalens. The L1 address
of `mailboxes.watcher.debug_ring_buf` is resolved from a precompiled brisc ELF
on disk (the layout of `mailboxes_t` is firmware-independent for a given arch).

The kernel writes:
    data[0]=ring_iter_0, data[1]=ring_id_0, data[2]=ring_iter_1, ...
    Pre-fill is 0xffffffff, so unwritten slots show as "??".

Usage:
    # from tt-metal repo root:
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
from dataclasses import dataclass

from ttexalens.tt_exalens_init import init_ttexalens
from ttexalens.tt_exalens_lib import read_from_device
from ttexalens.server import FileAccessApi
from ttexalens.elf.parsed import read_elf
from ttexalens.memory_access import FixedMemoryAccess


@dataclass
class RingBufLayout:
    base: int  # L1 address of debug_ring_buf
    data_offset: int  # offset of data[] within debug_ring_buf
    elements: int  # DEBUG_RING_BUFFER_ELEMENTS
    total_size: int  # bytes to read (header + data)


def find_brisc_elf(repo_root: str) -> str:
    candidates: list[str] = []
    candidates += glob.glob(os.path.join(repo_root, "tt_metal/pre-compiled/*/brisc/brisc.elf"))
    candidates += glob.glob(os.path.join(repo_root, "built/*/firmware/brisc/brisc.elf"))
    candidates += glob.glob(os.path.join(repo_root, "built/firmware/brisc/brisc.elf"))
    if not candidates:
        raise FileNotFoundError(
            "Could not auto-locate a brisc.elf. Pass --elf explicitly to a brisc.elf "
            "(any one will do; mailboxes_t layout is firmware-independent)."
        )
    # Pick the most recently modified one.
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def resolve_layout(elf_path: str) -> RingBufLayout:
    parsed = read_elf(FileAccessApi(), elf_path)
    fake_mem = FixedMemoryAccess(b"\x00" * 4096)
    mailboxes = parsed.get_global("mailboxes", fake_mem)
    rb = mailboxes.watcher.debug_ring_buf
    base = rb.get_address()
    data_offset = rb.data.get_address() - base
    elements = parsed.get_constant("DEBUG_RING_BUFFER_ELEMENTS")
    total_size = data_offset + elements * 4
    return RingBufLayout(base=base, data_offset=data_offset, elements=int(elements), total_size=total_size)


def parse_pairs(buf: bytes, layout: RingBufLayout) -> tuple[int, int, list[tuple[str, str]]]:
    # Header: int16 current_ptr, uint16 wrapped, then data[].
    current_ptr = int.from_bytes(buf[0:2], "little", signed=True)
    wrapped = int.from_bytes(buf[2:4], "little", signed=False)
    if current_ptr == -1:
        return current_ptr, wrapped, []
    data_bytes = buf[layout.data_offset : layout.data_offset + layout.elements * 4]
    words = [int.from_bytes(data_bytes[i * 4 : (i + 1) * 4], "little") for i in range(layout.elements)]

    # If the kernel is the only writer and pre-filled with 0xffffffff before
    # resetting current_ptr, then values written so far live at indices
    # [0 .. current_ptr] in physical order (no wrap to worry about up to 32 entries).
    if not wrapped:
        written = words[: current_ptr + 1]
    else:
        # If somebody wrapped, walk newest-first and reverse.
        order = []
        idx = current_ptr
        for _ in range(layout.elements):
            order.append(words[idx])
            idx = layout.elements - 1 if idx == 0 else idx - 1
        order.reverse()
        written = order

    def fmt(v: int) -> str:
        return "??" if v == 0xFFFFFFFF else str(v)

    pairs: list[tuple[str, str]] = []
    for i in range(0, len(written), 2):
        ri = fmt(written[i])
        rid = fmt(written[i + 1]) if i + 1 < len(written) else "<missing>"
        pairs.append((ri, rid))
    return current_ptr, wrapped, pairs


def main() -> int:
    ap = argparse.ArgumentParser(description="Dump SDPA ring_iter/ring_id pairs from watcher ring buffer (PCIe-only)")
    ap.add_argument("--elf", help="Path to any brisc.elf (auto-detected from tt_metal/pre-compiled if omitted)")
    ap.add_argument(
        "--devices",
        help="Comma-separated device ids to scan (default: all visible to tt-exalens)",
    )
    ap.add_argument("--csv", action="store_true", help="Emit CSV: device_id,core,ring_iter,ring_id,pair_index")
    ap.add_argument("--show-empty", action="store_true", help="Print rows for cores whose ring buffer is empty")
    args = ap.parse_args()

    repo_root = os.environ.get(
        "TT_METAL_HOME", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    elf_path = args.elf or find_brisc_elf(repo_root)
    print(f"# using ELF: {elf_path}", file=sys.stderr)
    layout = resolve_layout(elf_path)
    print(
        f"# mailboxes.watcher.debug_ring_buf @ 0x{layout.base:x}, "
        f"data offset=0x{layout.data_offset:x}, elements={layout.elements}",
        file=sys.stderr,
    )

    context = init_ttexalens()
    if args.devices:
        device_ids = [int(x) for x in args.devices.split(",") if x.strip()]
    else:
        device_ids = sorted(int(d) for d in context.devices.keys())
    print(f"# scanning {len(device_ids)} device(s): {device_ids}", file=sys.stderr)

    if args.csv:
        print("device_id,core,pair_index,ring_iter,ring_id")

    for did in device_ids:
        device = context.devices[did]
        try:
            cores = device.get_block_locations(block_type="functional_workers")
        except Exception as e:
            print(f"# device {did}: cannot enumerate cores ({e})", file=sys.stderr)
            continue
        for core in cores:
            try:
                buf = read_from_device(core, layout.base, device_id=did, num_bytes=layout.total_size, context=context)
            except Exception as e:
                print(f"# dev {did} {core}: read failed ({e})", file=sys.stderr)
                continue
            current_ptr, wrapped, pairs = parse_pairs(buf, layout)
            core_str = str(core)
            if not pairs:
                if args.show_empty:
                    if args.csv:
                        print(f"{did},{core_str},,,")
                    else:
                        print(f"dev={did} core={core_str} <empty> (current_ptr={current_ptr})")
                continue
            if args.csv:
                for i, (ri, rid) in enumerate(pairs):
                    print(f"{did},{core_str},{i},{ri},{rid}")
            else:
                pretty = " ".join(f"({ri},{rid})" for ri, rid in pairs)
                wrap_tag = " WRAPPED" if wrapped else ""
                print(f"dev={did} core={core_str} ptr={current_ptr}{wrap_tag} pairs={pretty}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
