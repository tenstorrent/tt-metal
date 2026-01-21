#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    check_core_magic.py

Description:
    This script checks if the core_magic_number in each core's mailbox matches
    the expected firmware type for that location. If there's a mismatch, it
    attempts to read the mailbox using other firmware types to identify what
    firmware is actually present.

Owner:
    jbaumanTT
"""

from ttexalens.context import Context
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.elf import ParsedElfFile
from ttexalens.memory_access import MemoryAccess
from dispatcher_data import run as get_dispatcher_data, DispatcherData
from elfs_cache import run as get_elfs_cache
from run_checks import run as get_run_checks
from triage import ScriptConfig, log_check_location, run_script

script_config = ScriptConfig(
    depends=["run_checks", "dispatcher_data", "elfs_cache"],
)

MAILBOX_CORRUPTED_MESSAGE = "Mailbox is likely corrupted, potentially due to NoC writes to an invalid location."


class CoreMagicValues:
    """
    Holds the magic values read from firmware ELF.
    Values are read once and cached for reuse.
    """

    def __init__(self, fw_elf):
        # Read CoreMagicNumber enum values from firmware
        self.worker = fw_elf.get_enum_value("CoreMagicNumber::WORKER")
        self.active_eth = fw_elf.get_enum_value("CoreMagicNumber::ACTIVE_ETH")
        self.idle_eth = fw_elf.get_enum_value("CoreMagicNumber::IDLE_ETH")

        self.magic_to_name = {
            self.worker: "WORKER",
            self.active_eth: "ACTIVE_ETH",
            self.idle_eth: "IDLE_ETH",
        }

    def get_name(self, magic_value: int) -> str | None:
        return self.magic_to_name.get(magic_value, None)


def get_expected_magic_for_location(location: OnChipCoordinate, magic_values: CoreMagicValues) -> tuple[int, str]:
    """
    Determine the expected magic number based on location type.
    Returns (magic_value, type_name).
    """
    block_type = location.noc_block.block_type
    if block_type == "functional_workers":
        return magic_values.worker, "WORKER"
    elif location in location.device.idle_eth_block_locations:
        return magic_values.idle_eth, "IDLE_ETH"
    elif location in location.device.active_eth_block_locations:
        return magic_values.active_eth, "ACTIVE_ETH"
    else:
        # Default to worker for unknown types
        return magic_values.worker, "WORKER"


def try_read_magic_with_dispatcher_data(
    location: OnChipCoordinate,
    risc_name: str,
    dispatcher_data: DispatcherData,
) -> int | None:
    """
    Attempt to read core_magic_number using the given firmware ELF.
    Returns the magic value or None if reading fails.
    """
    try:
        dispatcher_core_data = dispatcher_data.get_cached_core_data(location, risc_name)
        return dispatcher_core_data.mailboxes.core_info.core_magic_number.read_value()
    except Exception as e:
        log_check_location(
            location,
            False,
            f"{risc_name}: Failed to read core_magic_number from mailbox: {e}",
        )
        return None


def try_read_magic_with_elf(
    l1_mem_access: MemoryAccess,
    fw_elf: ParsedElfFile,
) -> int | None:
    """
    Attempt to read core_magic_number using the given firmware ELF.
    Returns the magic value or None if reading fails.
    """
    return fw_elf.get_global("mailboxes", l1_mem_access).core_info.core_magic_number.read_value()


def check_core_magic(
    location: OnChipCoordinate,
    risc_name: str,
    dispatcher_data: DispatcherData,
    magic_values: CoreMagicValues,
):
    """
    Check if the core_magic_number matches the expected firmware type.
    If mismatch, try other firmware types to identify what's actually present.
    """
    expected_magic, expected_type = get_expected_magic_for_location(location, magic_values)

    # Read the magic number from the expected mailbox location
    actual_magic = try_read_magic_with_dispatcher_data(location, risc_name, dispatcher_data)

    if actual_magic is None:
        return

    # Check if magic matches expected
    if actual_magic == expected_magic:
        log_check_location(
            location,
            True,
            f"{risc_name}: core_magic_number OK ({expected_type})",
        )
        return

    # Unknown magic at expected location - try other firmware ELFs to find a match
    other_elfs_to_try = []

    # Add the firmware ELFs we haven't tried yet
    if expected_type != "WORKER":
        other_elfs_to_try.append(("WORKER", dispatcher_data._brisc_elf))
    if expected_type != "IDLE_ETH":
        other_elfs_to_try.append(("IDLE_ETH", dispatcher_data._idle_erisc_elf))
    if expected_type != "ACTIVE_ETH":
        other_elfs_to_try.append(("ACTIVE_ETH", dispatcher_data._active_erisc_elf))

    found_type = None
    for type_name, other_elf in other_elfs_to_try:
        l1_mem_access = MemoryAccess.create_l1(location)
        other_magic = try_read_magic_with_elf(l1_mem_access, other_elf)
        if other_magic is not None:
            other_type_name = magic_values.get_name(other_magic)
            if other_type_name == type_name:
                found_type = type_name
                break

    if found_type:
        log_check_location(
            location,
            False,
            f"{risc_name}: core_magic_number mismatch! Expected {expected_type} (0x{expected_magic:08X}), "
            f"but found {found_type} firmware at {found_type} mailbox location. "
            f"Value at expected location: 0x{actual_magic:08X}. Triage may have incorrectly identified the firmware type.",
        )
    else:
        log_check_location(
            location,
            False,
            f"{risc_name}: core_magic_number mismatch! Expected {expected_type} (0x{expected_magic:08X}), "
            f"found unknown value 0x{actual_magic:08X}. Could not identify firmware type at other locations. {MAILBOX_CORRUPTED_MESSAGE}",
        )


def run(args, context: Context):
    BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth", "active_eth"]
    # Only check one RISC per core since magic is core-wide, not per-RISC
    # Use brisc for tensix, erisc/erisc0 for eth
    RISC_CORES_TO_CHECK = ["brisc", "erisc", "erisc0"]

    dispatcher_data = get_dispatcher_data(args, context)
    elfs_cache = get_elfs_cache(args, context)
    run_checks = get_run_checks(args, context)

    # Read magic values from firmware (identical across all firmware types)
    magic_values = CoreMagicValues(dispatcher_data._brisc_elf)

    run_checks.run_per_core_check(
        lambda location, risc_name: check_core_magic(location, risc_name, dispatcher_data, magic_values),
        block_filter=BLOCK_TYPES_TO_CHECK,
        core_filter=RISC_CORES_TO_CHECK,
    )


if __name__ == "__main__":
    run_script()
