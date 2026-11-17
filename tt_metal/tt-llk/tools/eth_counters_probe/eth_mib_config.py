#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reads the two Blackhole MAC configuration registers that decide whether the MAC statistics counters can be read
from a host without disturbing the base firmware, and prints the bits. Reads two registers per ethernet core, writes
nothing, and does not touch a statistics counter or a snapshot trigger.

  python eth_mib_config.py

TX configuration register (0xFFBA2200): bit 5 snapshot clears the active counters, bit 6 periodic snapshots, bit 7
active counters clear on read, bit 8 snapshot copies clear on read. RX configuration register (0xFFBA2400): bit 7
snapshot clears the active counters, bit 8 periodic snapshots, bit 9 active counters clear on read, bit 10 snapshot
copies clear on read.

A copy can be read without side effect only when its clear-on-read bit is 0; the active counters also move backwards
at every snapshot while the snapshot-clear bit is 1, so a host delta over them is only valid if no snapshot happens
inside the window (periodic bit 0 and the firmware not triggering).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as c  # noqa: E402

TX_BITS = {
    "snapshot clears active": 5,
    "periodic snapshot": 6,
    "active clears on read": 7,
    "snapshot copy clears on read": 8,
}
RX_BITS = {
    "snapshot clears active": 7,
    "periodic snapshot": 8,
    "active clears on read": 9,
    "snapshot copy clears on read": 10,
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default=c.default_device_path())
    args = p.parse_args()
    if c.card_busy(args.device):
        c.fail(f"{args.device} is open by another process; not running")
    cards = c.Cards()
    for device in cards.devices:
        if cards.arch(device) != "blackhole":
            print(f"device {device.id}: not Blackhole, skipped")
            continue
        for core in cards.eth_cores(device):
            if c.link_up("blackhole", core) is False:
                continue
            for label, address, bits in (
                ("TX", c.MAC_TX_CFG, TX_BITS),
                ("RX", c.MAC_RX_CFG, RX_BITS),
            ):
                word = core.noc_read32(address)
                state = ", ".join(f"{k} {(word >> b) & 1}" for k, b in bits.items())
                print(
                    f"device {device.id} core {core.to_str('noc0')} {label} cfg 0x{word:08x}: {state}"
                )


if __name__ == "__main__":
    main()
