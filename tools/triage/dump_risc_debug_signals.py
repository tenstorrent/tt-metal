#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_risc_debug_signals

Description:
    Goes through all tensix and idle_eth blocks and tries to halt every core inside them.
    If halt is successful, does nothing. If it throws an exception, prints all debug bus
    signals related to that core (e.g., for brisc, prints all signals matching brisc*).
"""

from dataclasses import dataclass
from triage import ScriptConfig, collection_serializer, triage_field, run_script
from run_checks import run as get_run_checks
from dispatcher_data import run as get_dispatcher_data, DispatcherData
from elfs_cache import run as get_elfs_cache, ElfsCache
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.context import Context
from ttexalens.tt_exalens_lib import read_words_from_device, write_words_to_device

script_config = ScriptConfig(
    depends=["run_checks", "dispatcher_data", "elfs_cache"],
    disabled=True,
)


@dataclass
class DumpDebugBusSignals:
    names: list[str] = triage_field("Debug Signals", collection_serializer("\n"))
    values: list[str] = triage_field("Values", collection_serializer("\n"))


def dump_risc_debug_signals(
    location: OnChipCoordinate, risc_name: str, dispatcher_data: DispatcherData, elfs_cache: ElfsCache
) -> DumpDebugBusSignals | None:
    """
    Try to halt a RISC core. If successful, return None.
    If it throws an exception, collect and return debug bus signals.
    """
    noc_block = location._device.get_block(location)

    try:
        risc_debug = noc_block.get_risc_debug(risc_name)
        # Try to halt the core
        risc_debug.halt()
        # If halt was successful, return None
        return None
    except:
        # If halt failed, collect debug bus signals
        debug_bus = noc_block.debug_bus
        if debug_bus is not None:
            # Filter groups that match the pattern risc_name*
            matching_groups = [group_name for group_name in debug_bus.group_names if group_name.startswith(risc_name)]

            if matching_groups:
                # We are using first 16 bytes of the firmware text section to collect debug bus signals
                dispatcher_core_data = dispatcher_data.get_cached_core_data(location, risc_name)
                firmware_elf = elfs_cache[dispatcher_core_data.firmware_path]
                firmware_text_address = firmware_elf.elf.get_section_by_name(".text")["sh_addr"]
                # Make sure that l1 address we are using is in the first 1MiB of the L1 cache (required for reading groups)
                assert firmware_text_address < 0x100000

                # Since we are rewritting the firmware text, we need to read the original data to restore it later
                original_data = read_words_from_device(location, firmware_text_address, word_count=4)

                l1_address = firmware_text_address

                # Collect all signals from all matching groups
                signal_names_str: list[str] = []
                signal_values_hex: list[str] = []
                for group_name in sorted(matching_groups):
                    # Read the signal group
                    group_sample = debug_bus.read_signal_group(group_name, l1_address)
                    # Iterate through all signals in the group
                    for signal_name in sorted(group_sample.keys()):
                        signal_names_str.append(f"{signal_name[len(risc_name)+1:]}")
                        signal_values_hex.append(f"{hex(group_sample[signal_name])}")

        # Restoring the original data
        write_words_to_device(location, firmware_text_address, original_data)
        # Verifying that the original data was restored
        assert read_words_from_device(location, firmware_text_address, word_count=4) == original_data

        return DumpDebugBusSignals(
            names=signal_names_str,
            values=signal_values_hex,
        )


def run(args, context: Context):
    BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth"]
    CORE_TYPES_TO_CHECK = ["brisc", "trisc0", "trisc1", "trisc2", "erisc", "erisc0", "erisc1"]

    run_checks = get_run_checks(args, context)
    dispatcher_data = get_dispatcher_data(args, context)
    elfs_cache = get_elfs_cache(args, context)

    return run_checks.run_per_core_check(
        lambda location, risc_name: dump_risc_debug_signals(location, risc_name, dispatcher_data, elfs_cache),
        block_filter=BLOCK_TYPES_TO_CHECK,
        core_filter=CORE_TYPES_TO_CHECK,
    )


if __name__ == "__main__":
    run_script()
