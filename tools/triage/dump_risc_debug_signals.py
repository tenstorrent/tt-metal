#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_risc_debug_signals [--path=<path>]

Description:
    Goes through all tensix and idle_eth blocks and tries to halt every RISC core inside them.
    If all cores halt successfully, does nothing. If any core fails to halt, collects all debug bus
    groups for that block. All collected data is written to a single JSON file.

Owner:
    adjordjevic-TT
"""

from collections import defaultdict
import json
import os
from triage import ScriptConfig, log_warning, run_script, log_check_location
from ttexalens.umd_device import TimeoutDeviceRegisterError
from run_checks import run as get_run_checks, RunChecks
from dispatcher_data import run as get_dispatcher_data, DispatcherData
from elfs_cache import run as get_elfs_cache, ElfsCache
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.context import Context
from ttexalens.tt_exalens_lib import read_words_from_device, write_words_to_device
from ttexalens.hardware.risc_debug import RiscHaltError

script_config = ScriptConfig(
    depends=["run_checks", "dispatcher_data", "elfs_cache"],
    disabled=os.getenv("TT_RUN_DISABLED_TRIAGE_SCRIPTS_IN_CI") is None,
)

# RISC cores to check for each block type
BLOCK_RISC_CORES = {
    "tensix": ["brisc", "trisc0", "trisc1", "trisc2"],
    "idle_eth": ["erisc", "erisc0", "erisc1"],
}


def get_firmware_text_address(
    location: OnChipCoordinate, risc_name: str, dispatcher_data: DispatcherData, elfs_cache: ElfsCache
) -> int:
    """Get the firmware text section address for a given RISC core."""
    dispatcher_core_data = dispatcher_data.get_cached_core_data(location, risc_name)
    firmware_elf = elfs_cache[dispatcher_core_data.firmware_path]
    firmware_text_address = firmware_elf.elf.get_section_by_name(".text")["sh_addr"]
    # Make sure that l1 address we are using is in the first 1MiB of the L1 cache (required for reading groups)
    assert firmware_text_address < 0x100000
    return firmware_text_address


def collect_debug_bus_signals(
    location: OnChipCoordinate, dispatcher_data: DispatcherData, elfs_cache: ElfsCache, run_checks: RunChecks
) -> dict | None:
    """
    Try to halt all RISC cores in a block. If all halt successfully, return None.
    If any core fails to halt, collect and return failed_riscs and debug_bus_signals.
    """
    device = location._device
    noc_block = device.get_block(location)
    block_type = run_checks.get_block_type(location)

    # Get available RISC cores for this block
    available_riscs = noc_block.risc_names
    riscs_to_check = [r for r in available_riscs if r in BLOCK_RISC_CORES.get(block_type, [])]

    # Try to halt all RISC cores, track which ones failed
    failed_riscs: list[str] = []
    for risc_name in riscs_to_check:
        risc_debug = noc_block.get_risc_debug(risc_name)
        if risc_debug.is_in_reset():
            continue

        try:
            with risc_debug.ensure_halted():
                pass
        except RiscHaltError:
            failed_riscs.append(risc_name)

    # If no cores failed to halt, nothing to collect
    if not failed_riscs:
        return None

    # At least one core failed to halt - collect all debug bus signals for this block
    debug_bus = noc_block.debug_bus
    all_groups = debug_bus.group_names

    # We are using first 16 bytes of the firmware text section to collect debug bus signals
    # Use the first failed risc to get the firmware text address
    risc_for_address = failed_riscs[0]
    l1_address = get_firmware_text_address(location, risc_for_address, dispatcher_data, elfs_cache)

    # Since we are rewriting the firmware text, we need to read the original data to restore it later
    original_data = read_words_from_device(location, l1_address, word_count=4)
    try:
        # Collect all debug bus groups as group_name -> 128-bit hex value
        debug_bus_data: dict[str, str] = {}
        for group_name in sorted(all_groups):
            # Read the signal group (this writes the 128-bit value to l1_address)
            group_sample = debug_bus.read_signal_group(group_name, l1_address)
            # Read the raw 128-bit value from l1_address
            debug_bus_data[group_name] = f"0x{group_sample.raw_data:032x}"

        # Return only the core data - location/device info comes from PerBlockCheckResult
        return {
            "failed_riscs": failed_riscs,
            "debug_bus_signal_groups": debug_bus_data,
        }
    except TimeoutDeviceRegisterError:
        raise
    except Exception as e:
        log_check_location(location, False, f"Failed to collect debug bus signals: {e}")
        return None
    finally:
        # Restoring the original data
        write_words_to_device(location, l1_address, original_data)
        # Verifying that the original data was restored
        assert read_words_from_device(location, l1_address, word_count=4) == original_data


def run(args, context: Context):
    BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth"]

    run_checks = get_run_checks(args, context)
    dispatcher_data = get_dispatcher_data(args, context)
    elfs_cache = get_elfs_cache(args, context)

    # Use RunChecks infrastructure to iterate over blocks
    results = run_checks.run_per_block_check(
        lambda location: collect_debug_bus_signals(location, dispatcher_data, elfs_cache, run_checks),
        block_filter=BLOCK_TYPES_TO_CHECK,
    )

    # Extract the collected data from wrapped results and write to single JSON file
    if results:
        all_debug_bus_data = defaultdict(dict)
        for r in results:
            if r.result is None:
                continue
            # Build full structure using info from PerBlockCheckResult wrapper
            device = r.device_description.device
            location = r.location
            block_type = run_checks.get_block_type(location)
            if block_type not in all_debug_bus_data[f"Device {device.id}"]:
                all_debug_bus_data[f"Device {device.id}"][block_type] = defaultdict(dict)
            all_debug_bus_data[f"Device {device.id}"][block_type][f"location: {location.to_user_str()}"] = {
                "failed_riscs": r.result["failed_riscs"],
                "debug_bus_signal_groups": r.result["debug_bus_signal_groups"],
            }

        if all_debug_bus_data:
            output_path = args["--path"] if args["--path"] else "debug_bus_signal_groups.json"
            with open(output_path, "w") as f:
                json.dump(all_debug_bus_data, f, indent=2)
            log_warning(f"Some riscs are broken. Generated JSON file with debug bus signals at {output_path}")

    return None


if __name__ == "__main__":
    run_script()
