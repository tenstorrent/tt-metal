#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_watcher_ringbuffer

Description:
    Dump watcher ring buffer contents for all cores, skipping cores with empty buffers. This ringbuffer can be written
    into by using the WATCHER_RING_BUFFER_PUSH macro in a kernel.

Owner:
    jbaumanTT
"""

from dataclasses import dataclass
from triage import ScriptConfig, triage_field, run_script
from run_checks import run as get_run_checks
from elfs_cache import run as get_elfs_cache, ElfsCache
from dispatcher_data import run as get_dispatcher_data, DispatcherData
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.context import Context


script_config = ScriptConfig(
    depends=["run_checks", "dispatcher_data", "elfs_cache"],
)


@dataclass
class DumpRingBufferData:
    proc: str = triage_field("Proc")
    debug_ring_buffer: list[str] | None = triage_field("debug_ring_buffer")


def read_ring_buffer(
    location: OnChipCoordinate,
    block_type: str,
    risc_name: str,
    dispatcher_data: DispatcherData,
    elf_cache: ElfsCache,
):
    """Read watcher ring buffer for the core. Returns None if ring buffer is empty or unreadable."""
    try:
        fw_path = dispatcher_data.get_cached_core_data(location, risc_name).firmware_path
    except Exception:
        return None

    fw_elf = elf_cache[fw_path]
    mailboxes = dispatcher_data.get_cached_core_data(location, risc_name).mailboxes

    current_ptr = mailboxes.watcher.debug_ring_buf.current_ptr
    if current_ptr == 65535:
        # Nothing pushed
        return None

    wrapped = mailboxes.watcher.debug_ring_buf.wrapped

    ring_elements = fw_elf.get_constant("DEBUG_RING_BUFFER_ELEMENTS")
    assert isinstance(ring_elements, int)

    values: list[str] = []
    idx = current_ptr
    for _ in range(ring_elements):
        val = mailboxes.watcher.debug_ring_buf.data[idx]
        values.append(f"0x{val:08X}")
        if idx == 0:
            if wrapped == 0:
                break
            idx = ring_elements - 1
        else:
            idx -= 1

    if not values:
        return None

    return DumpRingBufferData(
        proc=block_type,
        debug_ring_buffer=values,
    )


def read_ring_buffer_for_block(
    location: OnChipCoordinate,
    dispatcher_data: DispatcherData,
    elf_cache: ElfsCache,
):
    """Select appropriate RISC per block type and read the ring buffer."""
    try:
        block_type = location._device.get_block_type(location)
    except Exception:
        return None

    risc_name = location.noc_block.risc_names[0]

    return read_ring_buffer(location, block_type, risc_name, dispatcher_data, elf_cache)


def run(args, context: Context):
    """Entry point for triage framework."""
    run_checks = get_run_checks(args, context)
    dispatcher_data = get_dispatcher_data(args, context)
    elfs_cache = get_elfs_cache(args, context)
    BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth", "active_eth"]
    return run_checks.run_per_block_check(
        lambda location: read_ring_buffer_for_block(location, dispatcher_data, elfs_cache),
        block_filter=BLOCK_TYPES_TO_CHECK,
    )


if __name__ == "__main__":
    run_script()
