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
from triage import ScriptConfig, log_check_risc, run_script, triage_field
from callstack_provider import (
    KernelCallstackWithMessage,
    format_callstack_with_message,
    run as get_callstack_provider,
    CallstackProvider,
)
from run_checks import run as get_run_checks
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.context import Context
from ttexalens.tt_exalens_lib import read_word_from_device
from utils import ORANGE, RST


script_config = ScriptConfig(
    depends=["run_checks", "callstack_provider"],
)


@dataclass
class LightweightAssertInfo:
    kernel_name: str | None = triage_field("Kernel Name")
    kernel_callstack_with_message: KernelCallstackWithMessage = triage_field(
        "Kernel Callstack", format_callstack_with_message
    )


def dump_callstacks(
    location: OnChipCoordinate,
    risc_name: str,
    callstack_provider: CallstackProvider,
) -> LightweightAssertInfo | None:
    try:
        risc_debug = location._device.get_block(location).get_risc_debug(risc_name)

        # We don't care about cores that are in reset
        if risc_debug.is_in_reset():
            return None

        # Define two instructions
        previous_instruction = None
        current_instruction = None

        # Check if PC is in code private memory (this only can be true for NCRISC on Wormhole)
        pc = risc_debug.get_pc()
        code_private_memory = risc_debug.get_code_private_memory()
        if code_private_memory is not None and code_private_memory.contains_private_address(pc):
            dispatcher_core_data = callstack_provider.dispatcher_data.get_core_data(location, risc_name)
            elf = callstack_provider.elfs_cache[dispatcher_core_data.kernel_path].elf
            text_section = elf.get_section_by_name(".text")
            if text_section is None or dispatcher_core_data.kernel_offset is None:
                return None
            data: bytes = text_section.data()
            address: int = dispatcher_core_data.kernel_offset
            offset = pc - address
            current_instruction = int.from_bytes(data[offset : offset + 4], "little")
            if offset >= 4:
                previous_instruction = int.from_bytes(data[offset - 4 : offset], "little")
        else:
            # Otherwise, read instructions directly from device
            current_instruction = read_word_from_device(location, pc)
            if pc >= 4:
                previous_instruction = read_word_from_device(location, pc - 4)

        # Check if core hit ebreak
        ebreak_instruction = 0x00100073
        rewind_pc_for_ebreak = False
        if previous_instruction == ebreak_instruction:
            rewind_pc_for_ebreak = True
        elif current_instruction != ebreak_instruction:
            return None

        callstack_data = callstack_provider.get_callstacks(location, risc_name, rewind_pc_for_ebreak)
        return LightweightAssertInfo(
            kernel_name=callstack_data.dispatcher_core_data.kernel_name,
            kernel_callstack_with_message=callstack_data.kernel_callstack_with_message,
        )

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
