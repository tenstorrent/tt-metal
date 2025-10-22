#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_watcher_ringbuffer

Description:
    Dump watcher ring buffer contents for all cores, skipping cores with empty buffers.
"""

from dataclasses import dataclass
from triage import ScriptConfig, triage_field, run_script
from run_checks import run as get_run_checks
from elfs_cache import run as get_elfs_cache, ElfsCache
from dispatcher_data import run as get_dispatcher_data, DispatcherData
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.context import Context
from ttexalens.firmware import ELF
from ttexalens.parse_elf import mem_access


script_config = ScriptConfig(
    depends=["run_checks", "dispatcher_data", "elfs_cache"],
)


@dataclass
class DumpRingBufferData:
    proc: str = triage_field("Proc")
    location: OnChipCoordinate = triage_field("Loc")
    debug_ring_buffer: list[str] | None = triage_field("debug_ring_buffer")


def _get_mem_reader(location: OnChipCoordinate, risc_name: str):
    """Get a mem_reader for the given RISC at the location, falling back to default if needed."""
    try:
        return ELF.get_mem_reader(location, risc_name)
    except Exception:
        return ELF.get_mem_reader(location)


def _read_scalar(elf_obj, expr: str, mem_reader):
    try:
        return int(mem_access(elf_obj, expr, mem_reader)[0][0])
    except Exception:
        return None


def _get_ring_elements(elf_obj, mem_reader) -> int:
    try:
        return int(mem_access(elf_obj, "DEBUG_RING_BUFFER_ELEMENTS", mem_reader)[3])
    except Exception:
        return 32


def read_ring_buffer(
    location: OnChipCoordinate,
    block_type: str,
    risc_name: str,
    dispatcher_data: DispatcherData,
    elf_cache: ElfsCache,
):
    """Read watcher ring buffer for the core. Returns None if ring buffer is empty or unreadable."""
    try:
        fw_path = dispatcher_data.get_core_data(location, risc_name).firmware_path
    except Exception:
        return None

    fw_elf = elf_cache[fw_path]
    mem_reader = _get_mem_reader(location, risc_name)

    current_ptr = _read_scalar(fw_elf, "mailboxes->watcher.debug_ring_buf.current_ptr", mem_reader)
    if current_ptr is None or current_ptr == -1:
        return None

    wrapped = _read_scalar(fw_elf, "mailboxes->watcher.debug_ring_buf.wrapped", mem_reader)
    if wrapped is None:
        wrapped = 0

    ring_elements = _get_ring_elements(fw_elf, mem_reader)

    values: list[str] = []
    idx = int(current_ptr)
    for _ in range(ring_elements):
        val = _read_scalar(
            fw_elf,
            f"mailboxes->watcher.debug_ring_buf.data[{idx}]",
            mem_reader,
        )
        if val is None:
            break
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
        location=location,
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

    # Map block type to the RISC to read from
    if block_type == "functional_workers":
        risc_name = "brisc"
    elif block_type == "idle_eth" or block_type == "active_eth":
        risc_name = "erisc"
    else:
        return None

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
