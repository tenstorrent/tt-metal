#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_device_state [--dump-halted-state]

Options:
    --dump-halted-state  Also halt every RISC that can be halted and store its GPRs and private memory. Cores stay halted.

Description:
    Stores the raw state of every Tensix and Ethernet block, so it can still be read after the device is reset. One
    row per debug-bus group, region, RISC GPR or RISC debug hardware register, from tt-exalens's read_block_state
    and, with the flag, halt_and_read_risc_state. Kind is NOC_REGION, PRIVATE_REGION, DEBUG_BUS_GROUP, GPR or
    DEBUG_HARDWARE_REGISTER. Data holds the raw bytes, or Reason says why there are none (for example "unsafe to
    read" or "in reset"). Address is the NOC address of a NOC region and the private address of a private region.
    A row without a Name is about a whole RISC.

    Rows go only to the dump_device_state table of the SQLite output, as each block is read, so the script needs
    triage's --sqlite-output-path; without it, the script reads nothing. It dumps the chips that triage's --dev
    option selects, remote chips before local chips, and finishes each chip before it starts the next one.
    A Blackhole chip takes about 260 MB and 50 s, or 270 s with the flag; an n300 (two Wormhole chips) takes about
    275 MB and 255 s, or 340 s with the flag.

    The script runs at low priority, after every other script, so a core that an earlier script halted is dumped
    halted. By default it halts no core. With --dump-halted-state it halts every core it can and leaves them halted.

    Note: This script is disabled by default. To enable it, set the environment variable:
        TT_TRIAGE_ENABLE_DEVICE_STATE_DUMP=1

Owner:
    onenezicTT
"""

import os
from dataclasses import dataclass

from run_checks import run as get_run_checks
from triage import (
    ScriptConfig,
    ScriptPriority,
    get_sqlite_serializer,
    hex_serializer,
    log_warning,
    run_script,
    triage_field,
)
from ttexalens.context import Context
from ttexalens.tt_exalens_lib import BlockStateEntryKind, halt_and_read_risc_state, read_block_state

script_config = ScriptConfig(
    depends=["run_checks"],
    priority=ScriptPriority.LOW,
    disabled=os.environ.get("TT_TRIAGE_ENABLE_DEVICE_STATE_DUMP") != "1",
)


@dataclass
class DeviceStateRow:
    kind: BlockStateEntryKind = triage_field("Kind", lambda kind: kind.name)
    risc_name: str | None = triage_field("RiscV")
    name: str | None = triage_field("Name")
    address: int | None = triage_field("Address", hex_serializer)
    size: int = triage_field("Size (bytes)")
    data: bytes | None = triage_field("Data", lambda data: "N/A" if data is None else f"{len(data)} bytes")
    reason: str | None = triage_field("Reason")


def run(args, context: Context):
    sqlite_serializer = get_sqlite_serializer()
    if sqlite_serializer is None:
        log_warning(
            "Nothing read: no SQLite output is open (--sqlite-output-path), and the device state is stored only there."
        )
        return
    run_checks = get_run_checks(args, context)
    reads = [read_block_state, halt_and_read_risc_state] if args["--dump-halted-state"] else [read_block_state]
    # Remote chips first: halting a local chip's link cores cuts off the chips reached through them.
    for device in sorted(run_checks.devices, key=lambda device: device.is_local):
        for read in reads:
            run_checks.run_per_block_check(
                lambda location, _read=read: [
                    DeviceStateRow(
                        kind=entry.kind,
                        risc_name=entry.risc_name,
                        name=entry.name,
                        address=entry.address,
                        size=entry.size,
                        data=entry.data,
                        reason=entry.reason,
                    )
                    for entry in _read(location)
                ],
                block_filter=["tensix", "eth"],
                devices=[device],
                store_block_rows=lambda block_rows: sqlite_serializer.insert_rows("dump_device_state", block_rows),
            )


if __name__ == "__main__":
    run_script()
