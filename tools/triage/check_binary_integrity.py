#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    check_binary_integrity [--dump-mismatch]

Options:
    --dump-mismatch  For each mismatched range, print its L1 addresses and the expected and found bytes.

Description:
    Checking that code binaries are the way they were uploaded to the device (both firmware and kernel).
    On a mismatch, it reports how many bytes differ, in how many ranges, and which kernel was expected.

Owner:
    tt-vjovanovic
"""

from dispatcher_data import run as get_dispatcher_data, DispatcherData
from elfs_cache import run as get_elfs_cache, ElfsCache
from run_checks import run as get_run_checks
import os
from ttexalens.context import Context
from ttexalens.hardware.risc_debug import RiscLocation
from ttexalens._native_ttexalens import MemoryAccess
from triage import ScriptConfig, log_check_risc, run_script

script_config = ScriptConfig(
    depends=["run_checks", "dispatcher_data", "elfs_cache"],
)


def find_mismatched_ranges(expected: memoryview | bytes, found: bytearray | bytes) -> list[tuple[int, int]]:
    """Return (offset, length) of each run of consecutive bytes that differ. Both buffers must have the same length."""
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for offset, (expected_byte, found_byte) in enumerate(zip(expected, found)):
        if expected_byte != found_byte:
            if start is None:
                start = offset
        elif start is not None:
            ranges.append((start, offset - start))
            start = None
    if start is not None:
        ranges.append((start, len(expected) - start))
    return ranges


def format_mismatch_message(
    section_name: str, address: int, size: int, ranges: list[tuple[int, int]], elf_path: str, kernel_name: str | None
) -> str:
    """One-line summary of a section whose bytes in L1 differ from the ELF. kernel_name is None for firmware."""
    mismatched_bytes = sum(length for _, length in ranges)
    range_word = "range" if len(ranges) == 1 else "ranges"
    message = (
        f"Data mismatch in {section_name} at 0x{address:08x}: "
        f"{mismatched_bytes}/{size} bytes differ in {len(ranges)} {range_word}."
    )
    if kernel_name is not None:
        message += f" Expected kernel: {kernel_name}."
    return f"{message} ELF: {elf_path}"


def format_mismatch_dump(
    ranges: list[tuple[int, int]], expected: memoryview | bytes, found: bytearray | bytes, load_address: int
) -> str:
    """Lines for --dump-mismatch: each range's L1 addresses, then its expected bytes and the bytes found in L1."""
    lines: list[str] = []
    for number, (offset, length) in enumerate(ranges, start=1):
        start_address = load_address + offset
        end_address = start_address + length - 1
        lines.append(f"      range {number}/{len(ranges)}: 0x{start_address:08x}-0x{end_address:08x} ({length} B)")
        lines.append(f"        expected: {expected[offset : offset + length].hex(' ')}")
        lines.append(f"        found:    {found[offset : offset + length].hex(' ')}")

    return "\n".join(lines)


def check_elf_integrity(
    risc_location: RiscLocation,
    l1_mem_access: MemoryAccess,
    elfs_cache: ElfsCache,
    elf_path: str,
    description: str,
    load_address: int | None,
    kernel_name: str | None,
    dump_mismatch: bool,
):
    """Log a check error for each verified section of the ELF whose bytes in L1 differ from the ELF."""
    location = risc_location.location
    risc_name = risc_location.risc_name
    elf_exists = os.path.exists(elf_path)
    log_check_risc(risc_name, location, elf_exists, f"{description} ELF file {elf_path} does not exist.")
    if not elf_exists:
        return

    elf_file = elfs_cache[elf_path]
    sections_to_verify = [".text"]
    for section_name in sections_to_verify:
        section = elf_file.get_section_by_name(section_name)
        if section is None:
            log_check_risc(risc_name, location, False, f"Section {section_name} not found in ELF file {elf_path}.")
            continue

        # Firmware is in L1 at the address in its ELF; a kernel is loaded at load_address
        address = section.address if load_address is None else load_address
        read_data = bytearray(len(section.data))
        l1_mem_access.read(address, read_data)
        if memoryview(read_data) == section.data:
            continue

        ranges = find_mismatched_ranges(section.data, read_data)
        message = format_mismatch_message(section_name, address, len(section.data), ranges, elf_path, kernel_name)
        if dump_mismatch:
            message += "\n" + format_mismatch_dump(ranges, section.data, read_data, address)
        log_check_risc(risc_name, location, False, message)


def check_binary_integrity(
    risc_location: RiscLocation, dispatcher_data: DispatcherData, elfs_cache: ElfsCache, dump_mismatch: bool
):
    location = risc_location.location
    risc_name = risc_location.risc_name
    if not dispatcher_data.risc_enabled(risc_name):
        return

    l1_mem_access = dispatcher_data.l1_memory_access(location)
    dispatcher_core_data = dispatcher_data.get_cached_core_data(risc_location)

    # Check firmware ELF binary state on the device
    check_elf_integrity(
        risc_location,
        l1_mem_access,
        elfs_cache,
        dispatcher_core_data.firmware_path,
        "Firmware",
        load_address=None,
        kernel_name=None,
        dump_mismatch=dump_mismatch,
    )

    # Check kernel ELF binary state on the device
    # Firmware clears enables after a run, so a disabled RISC's launch slot is stale
    if dispatcher_core_data.kernel_xip_path is not None and dispatcher_core_data.risc_enabled_by_kernel is not False:
        if dispatcher_core_data.kernel_offset is None:
            log_check_risc(
                risc_name,
                location,
                False,
                f"Kernel offset not set for ELF file {dispatcher_core_data.kernel_xip_path}.",
            )
        else:
            check_elf_integrity(
                risc_location,
                l1_mem_access,
                elfs_cache,
                dispatcher_core_data.kernel_xip_path,
                "Kernel",
                load_address=dispatcher_core_data.kernel_offset,
                kernel_name=dispatcher_core_data.kernel_name,
                dump_mismatch=dump_mismatch,
            )


def run(args, context: Context):
    BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth", "dram"]
    dump_mismatch = bool(args["--dump-mismatch"])
    dispatcher_data = get_dispatcher_data(args, context)
    elfs_cache = get_elfs_cache(args, context)
    run_checks = get_run_checks(args, context)
    run_checks.run_per_core_check(
        lambda risc_debug: check_binary_integrity(risc_debug.risc_location, dispatcher_data, elfs_cache, dump_mismatch),
        block_filter=BLOCK_TYPES_TO_CHECK,
    )


if __name__ == "__main__":
    run_script()
