#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Read N words from a given L1 address on every tensix worker core, on every
device visible to tt-exalens. Reads happen over PCIe — no live host process
or inspector logs required.

Usage:
    python tools/triage/read_noc_all_cores.py <addr> <num_words>
    python tools/triage/read_noc_all_cores.py 0x4c4 32
    python tools/triage/read_noc_all_cores.py 0x4c4 32 --devices 0,1,2
"""

from __future__ import annotations

import argparse
import sys

from ttexalens.tt_exalens_init import init_ttexalens
from ttexalens.tt_exalens_lib import read_words_from_device


def main() -> int:
    ap = argparse.ArgumentParser(description="Read N words from <addr> on all cores of all devices")
    ap.add_argument("addr", help="L1 address (decimal or 0x-prefixed hex)")
    ap.add_argument("num_words", type=int, help="Number of 4-byte words to read")
    ap.add_argument("--devices", help="Comma-separated device ids (default: all visible)")
    args = ap.parse_args()

    addr = int(args.addr, 0)
    num_words = args.num_words

    context = init_ttexalens()
    if args.devices:
        device_ids = [int(x) for x in args.devices.split(",") if x.strip()]
    else:
        device_ids = sorted(int(d) for d in context.devices.keys())

    for did in device_ids:
        device = context.devices[did]
        cores = device.get_block_locations(block_type="functional_workers")
        for core in cores:
            print(f"reading device {did}, core {core}:")
            words = read_words_from_device(core, addr, device_id=did, word_count=num_words, context=context)
            for i, w in enumerate(words):
                print(f"  [{i:2d}] 0x{w:08x}  ({w})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
