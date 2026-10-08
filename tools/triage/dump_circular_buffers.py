#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_circular_buffers [--dump-cb-content]

Options:
    --dump-cb-content   Read each CB's whole FIFO into the Content column, as hex. Only shown on -vv.

Description:
    Circular buffer state of every running Tensix core, one row per CB of the current program.

    The launch message's local_cb_mask says which CBs exist, and their geometry is the host-written
    config in the kernel config buffer. Kind is local for a CB whose producer and consumer are on this
    core, or global sender / global receiver for this core's end of a GlobalCircularBuffer. A receiver
    usually also has a local CB over the same memory; that is a separate queue, not a duplicate.

    Pushed/Popped count pages pushed by the producer and popped by the consumer, and Occupancy is the
    difference. Local counters are stream registers, reset for every program and compared modulo 2^16
    like the kernels do. Global counters live in L1 for the buffer's lifetime, and a global sender
    shows its slowest receiver.

    Content shows only at -vv, but --sqlite-output-path always records it. Global senders get none:
    they push from their local CB straight into the receivers.

    Read/write positions aren't read: they live in RISC private memory, and reading it halts the
    core. For a local CB they follow from the counters: rd = Address + (Popped * Page Size) % Size,
    and wr likewise from Pushed. That breaks for a local CB aliased onto a global one, for kernels
    that move CB pointers by hand, and after the 16-bit counter wraps.

Owner:
    onenezicTT
"""

import struct
from dataclasses import dataclass

from dispatcher_data import DispatcherData, run as get_dispatcher_data
from run_checks import run as get_run_checks
from triage import ScriptConfig, hex_serializer, run_script, triage_field
from ttexalens.context import Context
from ttexalens.hardware.risc_debug import RiscLocation
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.tt_exalens_lib import read_from_device, read_word_from_device

script_config = ScriptConfig(
    depends=["run_checks", "dispatcher_data"],
)

# (NUM_CIRCULAR_BUFFERS, tiles_received register, tiles_acked register); CB i uses overlay stream i.
Arch = tuple[int, int, int]
WORMHOLE: Arch = (32, 4, 3)
BLACKHOLE: Arch = (64, 10, 8)
STREAM_REG_BASE = 0xFFB40000
STREAM_REG_STRIDE = 0x1000
L1_ALIGNMENT = 16  # global CB counters count L1_ALIGNMENT-byte units


@dataclass
class CircularBufferRow:
    cb: int = triage_field("CB")
    kind: str = triage_field("Kind")
    address: int = triage_field("Address", hex_serializer)
    size: int = triage_field("Size")
    page_size: int = triage_field("Page Size")
    size_pages: int = triage_field("Size (pages)")
    pushed: int = triage_field("Pushed")
    popped: int = triage_field("Popped")
    occupancy: int = triage_field("Occupancy")
    content: str | None = triage_field("Content", verbose=2)


def read_words(location: OnChipCoordinate, address: int, count: int) -> tuple[int, ...]:
    return struct.unpack(f"<{count}I", read_from_device(location, address, num_bytes=4 * count))


def read_content(location: OnChipCoordinate, address: int, size: int) -> str:
    return read_from_device(location, address, num_bytes=size).hex()


def local_rows(
    location: OnChipCoordinate, kernel_config, config_base: int, arch: Arch, dump_content: bool
) -> list[CircularBufferRow]:
    num_cbs, received_reg, acked_reg = arch
    mask = int(kernel_config.local_cb_mask)
    base = config_base + int(kernel_config.local_cb_offset)
    rows = []
    for cb in range(num_cbs):
        if not mask >> cb & 1:
            continue
        address, size, pages, page_size = read_words(location, base + 16 * cb, 4)
        stream = STREAM_REG_BASE + STREAM_REG_STRIDE * cb
        pushed = read_word_from_device(location, stream + 4 * received_reg)
        popped = read_word_from_device(location, stream + 4 * acked_reg)
        occupancy = (pushed - popped) & 0xFFFF
        content = read_content(location, address, size) if dump_content else None
        rows.append(CircularBufferRow(cb, "local", address, size, page_size, pages, pushed, popped, occupancy, content))
    return rows


def read_counters(location: OnChipCoordinate, address: int) -> tuple[int, int]:
    # pages_sent, and pages_acked one L1_ALIGNMENT later
    return read_word_from_device(location, address), read_word_from_device(location, address + L1_ALIGNMENT)


def remote_rows(
    location: OnChipCoordinate, kernel_config, config_base: int, arch: Arch, dump_content: bool
) -> list[CircularBufferRow]:
    num_cbs = arch[0]
    base = config_base + int(kernel_config.remote_cb_offset)
    rows = []
    for cb in range(int(kernel_config.min_remote_cb_start_index), num_cbs):
        # Remote configs are laid out from the last CB index down.
        config_addr, page_size = read_words(location, base + 8 * (num_cbs - 1 - cb), 2)
        if config_addr == 0:
            continue
        is_sender, num_receivers, address, size, _, _, counters, _ = read_words(location, config_addr, 8)
        # A sender has a counter pair per receiver, a receiver only its own; keep the one furthest behind.
        slots = num_receivers if is_sender else 1
        pairs = [read_counters(location, counters + 2 * L1_ALIGNMENT * i) for i in range(slots)]
        sent, acked = max(pairs, key=lambda pair: (pair[0] - pair[1]) & 0xFFFFFFFF)
        units = page_size // L1_ALIGNMENT  # counter units per page
        pushed, popped = sent // units, acked // units
        occupancy = ((sent - acked) & 0xFFFFFFFF) // units
        kind = "global sender" if is_sender else "global receiver"
        pages = size // page_size
        content = read_content(location, address, size) if dump_content and not is_sender else None
        rows.append(CircularBufferRow(cb, kind, address, size, page_size, pages, pushed, popped, occupancy, content))
    return rows


def read_core(
    location: OnChipCoordinate, dispatcher_data: DispatcherData, dump_content: bool
) -> list[CircularBufferRow] | None:
    arch = WORMHOLE if location.device.is_wormhole() else BLACKHOLE if location.device.is_blackhole() else None
    if arch is None:
        return None
    core = dispatcher_data.get_cached_core_data(RiscLocation(location, None, "brisc"))
    if core.go_message == "DONE" or core.mailboxes is None:
        return None
    kernel_config = core.mailboxes.launch[core.launch_msg_rd_ptr].kernel_config
    if int(kernel_config.enables) == 0:
        return None
    rows = local_rows(location, kernel_config, core.kernel_config_base, arch, dump_content)
    rows += remote_rows(location, kernel_config, core.kernel_config_base, arch, dump_content)
    return rows or None


def run(args, context: Context):
    run_checks = get_run_checks(args, context)
    dispatcher_data = get_dispatcher_data(args, context)
    dump_content = bool(args["--dump-cb-content"])
    return run_checks.run_per_block_check(
        lambda location: read_core(location, dispatcher_data, dump_content), block_filter="tensix"
    )


if __name__ == "__main__":
    run_script()
