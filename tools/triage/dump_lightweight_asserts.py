#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_lightweight_asserts

Description:
    Dumps information about lightweight asserts that have occurred on the device.
"""

from dataclasses import dataclass

from triage import ScriptConfig, log_check_risc, run_script
from ttexalens import util
from callstack_provider import run as get_callstack_provider, CallstackProvider, CallstacksData
from run_checks import run as get_run_checks
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.context import Context
from ttexalens.tt_exalens_lib import read_word_from_device
from utils import ORANGE, RST


script_config = ScriptConfig(
    depends=["run_checks", "callstack_provider"],
)


def dump_callstacks(
    location: OnChipCoordinate,
    risc_name: str,
    callstack_provider: CallstackProvider,
) -> CallstacksData | None:
    try:
        risc_debug = location._device.get_block(location).get_risc_debug(risc_name)

        # We don't care about cores that are in reset
        if risc_debug.is_in_reset():
            return None

        # We cannot read NCRISC private memory to verify ebreak hit
        pc = risc_debug.get_pc()
        if pc >= 0xFFB00000:
            return None
        if pc < 4:
            return None

        # Check if core hit ebreak
        rewind_pc_for_ebreak = False
        if read_word_from_device(location, pc - 4) == 0x00100073:
            rewind_pc_for_ebreak = True
        elif read_word_from_device(location, pc) != 0x00100073:
            return None

        return callstack_provider.get_callstacks(location, risc_name, rewind_pc_for_ebreak)

    except Exception as e:
        log_check_risc(
            risc_name,
            location,
            False,
            f"{ORANGE}Failed to dump callstacks: {e}{RST}",
        )
        return None


def run(args, context: Context):
    BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth", "active_eth"]

    run_checks = get_run_checks(args, context)
    callstack_provider = get_callstack_provider(args, context)

    callstacks_data = run_checks.run_per_core_check(
        lambda location, risc_name: dump_callstacks(
            location,
            risc_name,
            callstack_provider,
        ),
        block_filter=BLOCK_TYPES_TO_CHECK,
    )

    return callstacks_data


if __name__ == "__main__":
    run_script()
