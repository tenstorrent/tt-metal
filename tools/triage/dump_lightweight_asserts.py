#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_lightweight_asserts

Description:
    Dumps information about lightweight asserts that have occurred on the device.

Owner:
    tt-vjovanovic
"""


import os
import re
from dataclasses import dataclass
from triage import ScriptConfig, log_check_risc, run_script, triage_field
from callstack_provider import (
    KernelCallstackWithMessage,
    format_callstack_with_message,
    serialize_variables,
    run as get_callstack_provider,
    CallstackProvider,
)
from dispatcher_data import run as get_dispatcher_data, DispatcherData
from run_checks import run as get_run_checks
from rich.markup import escape
from ttexalens.hardware.risc_debug import RiscDebug
from ttexalens.context import Context
from ttexalens.tt_exalens_lib import read_word_from_device
from ttexalens.elf import CallstackEntry, CallstackEntryVariable
from ttexalens.umd_device import TimeoutDeviceRegisterError


script_config = ScriptConfig(
    depends=["run_checks", "callstack_provider"],
)


@dataclass
class LightweightAssertCallstackWithCode:
    code_line: str
    callstack: KernelCallstackWithMessage


def format_callstack_with_message_and_callstack(callstack_with_message: LightweightAssertCallstackWithCode) -> str:
    return f"\n{callstack_with_message.code_line}\n{format_callstack_with_message(callstack_with_message.callstack)}"


@dataclass
class LightweightAssertInfo:
    kernel_name: str | None = triage_field("Kernel Name")
    kernel_callstack_with_message: LightweightAssertCallstackWithCode = triage_field(
        "Assert line with callstack", format_callstack_with_message_and_callstack
    )
    arguments_and_locals: str | None = triage_field("Arguments and Locals")


def _macro_name_start_for_needle(code_line: str, needle_index: int) -> int:
    """Move left from the open of a macro to the first character of its name (e.g. L in LLK_ASSERT)."""
    macro_start = needle_index
    while macro_start > 0 and (code_line[macro_start - 1].isalnum() or code_line[macro_start - 1] == "_"):
        macro_start -= 1
    return macro_start


def extract_assert_code(file: str | None, line: int | None, column: int | None) -> str:
    if file is None or line is None:
        return "?"

    if not os.path.exists(file):
        return "?file not found?"
    try:
        with open(file, "r") as f:
            lines = f.readlines()
            if not (0 <= line - 1 < len(lines)):
                return "?wrong line number? Check the first code line in the stack trace."
            code_line = lines[line - 1]

            start_index = -1

            def scan_for_needle(needle: str) -> None:
                nonlocal start_index
                search_from = 0
                while True:
                    new_index = code_line.find(needle, search_from)
                    if new_index == -1:
                        break
                    macro_start = _macro_name_start_for_needle(code_line, new_index)
                    if column is not None:
                        if macro_start == column:
                            start_index = new_index
                            break
                        if macro_start > column:
                            # Preserve the last ASSERT-like macro before the reported column.
                            # If none was seen yet, fall back to the first later macro so we
                            # still avoid returning "ASSERT() not found!" for approximate columns.
                            if start_index == -1:
                                start_index = new_index
                            break
                    start_index = new_index
                    search_from = new_index + 1

            scan_for_needle("ASSERT(")
            if start_index == -1:
                return "ASSERT() not found! Check the first code line in the stack trace."
            while start_index > 0 and (code_line[start_index - 1].isalnum() or code_line[start_index - 1] == "_"):
                start_index -= 1
            # Find the matching closing parenthesis for ASSERT(
            open_paren_index = code_line.find("(", start_index)
            if open_paren_index == -1:
                return "ASSERT() not opened! Check the first code line in the stack trace."
            paren_count = 1
            i = open_paren_index + 1
            while i < len(code_line):
                if code_line[i] == "(":
                    paren_count += 1
                elif code_line[i] == ")":
                    paren_count -= 1
                    if paren_count == 0:
                        break
                i += 1
            if paren_count != 0:
                current_line_idx = line
                while paren_count > 0 and current_line_idx < len(lines):
                    next_line = lines[current_line_idx]
                    code_line += " " + next_line.strip()
                    for ch in next_line:
                        if ch == "(":
                            paren_count += 1
                        elif ch == ")":
                            paren_count -= 1
                            if paren_count == 0:
                                break
                    current_line_idx += 1
                if paren_count != 0:
                    return "ASSERT() closing paren not found. Check the first code line in the stack trace."
                # Re-scan the assembled string to find the closing paren position
                i = open_paren_index + 1
                scan_count = 1
                while i < len(code_line):
                    if code_line[i] == "(":
                        scan_count += 1
                    elif code_line[i] == ")":
                        scan_count -= 1
                        if scan_count == 0:
                            break
                    i += 1
            return code_line[start_index : i + 1].strip()
    except Exception:
        return "?"


ASSERT_TRAP_FUNCTIONS = {"lightweight_assert_trap", "assert_and_hang"}


def _is_assert_trap_frame(frame: CallstackEntry) -> bool:
    return frame.function_name in ASSERT_TRAP_FUNCTIONS


def _find_assert_frame(callstack: list[CallstackEntry]) -> CallstackEntry:
    # The trap, when there is one, is the top frame and the ASSERT is the frame right below it.
    if len(callstack) > 1 and _is_assert_trap_frame(callstack[0]):
        return callstack[1]
    return callstack[0]


def dump_lightweight_asserts(
    risc_debug: RiscDebug,
    dispatcher_data: DispatcherData,
    callstack_provider: CallstackProvider,
) -> LightweightAssertInfo | None:
    risc_name = risc_debug.risc_location.risc_name
    location = risc_debug.risc_location.location
    try:
        if not dispatcher_data.risc_enabled(risc_name):
            return None

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
            dispatcher_core_data = callstack_provider.dispatcher_data.get_cached_core_data(risc_debug.risc_location)
            if dispatcher_core_data.kernel_path is None:
                return None
            elf = callstack_provider.elfs_cache[dispatcher_core_data.kernel_path]
            text_section = elf.get_section_by_name(".text")
            if text_section is None or dispatcher_core_data.kernel_offset is None:
                return None
            data = text_section.data
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

        # Check if core hit ebreak or is spinning in a while(true) loop after an assert failure
        while_true_instruction = 0x0000006F
        ebreak_instruction = 0x00100073
        rewind_pc_for_ebreak = False
        if previous_instruction == ebreak_instruction:
            rewind_pc_for_ebreak = True
        elif current_instruction != ebreak_instruction and current_instruction != while_true_instruction:
            return None

        callstack_data = callstack_provider.get_cached_callstacks(
            risc_debug, rewind_pc_for_ebreak, use_full_callstack=True
        )
        arguments_and_locals = None
        assert_code = "?"
        if callstack_data.kernel_callstack_with_message.callstack:
            assert_frame = _find_assert_frame(callstack_data.kernel_callstack_with_message.callstack)
            fi = assert_frame.file_info
            assert_code = extract_assert_code(
                fi.file if fi is not None else None,
                fi.line if fi is not None else None,
                fi.column if fi is not None else None,
            )
            arguments_and_locals = ""
            if len(assert_frame.template_parameters) > 0:
                arguments_and_locals += "\nTemplate parameters:\n"
                arguments_and_locals += serialize_variables(assert_frame.template_parameters, assert_code)
            if len(assert_frame.arguments) > 0:
                arguments_and_locals += "\nRuntime arguments:\n"
                arguments_and_locals += serialize_variables(assert_frame.arguments, assert_code)
            if len(assert_frame.locals) > 0:
                arguments_and_locals += "\nLocals:\n"
                arguments_and_locals += serialize_variables(assert_frame.locals, assert_code)
            names = {
                var.name
                for var in (*assert_frame.template_parameters, *assert_frame.arguments, *assert_frame.locals)
                if var.name
            }
            assert_code = re.sub(
                r"(?<!\\)\b\w+",  # whole words that don't start with a backslash
                lambda word: f"[info]{word.group()}[/]" if word.group() in names else word.group(),
                escape(assert_code),  # so brackets in the source (buf[i]) are not read as Rich tags
            )
            if arguments_and_locals:

                def _any_value_missing(vs: list[CallstackEntryVariable]) -> bool:
                    return any(v.value is None for v in vs)

                if any(
                    _any_value_missing(vs)
                    for vs in (
                        assert_frame.template_parameters,
                        assert_frame.arguments,
                        assert_frame.locals,
                    )
                ):
                    arguments_and_locals += (
                        "\n\n[dim]Values shown as '?' were not returned by the debugger "
                        "(e.g. optimized out, or not in a live range at the ebreak). The source may still be correct.[/]"
                    )
        return LightweightAssertInfo(
            kernel_name=callstack_data.dispatcher_core_data.kernel_name,
            kernel_callstack_with_message=LightweightAssertCallstackWithCode(
                assert_code, callstack_data.kernel_callstack_with_message
            ),
            arguments_and_locals=arguments_and_locals,
        )

    except TimeoutDeviceRegisterError:
        raise
    except Exception as e:
        log_check_risc(
            risc_name,
            location,
            False,
            f"[warning]Failed to dump lightweight asserts: {e}[/]",
        )
        return None


def run(args, context: Context):
    BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth", "active_eth", "dram"]

    run_checks = get_run_checks(args, context)
    callstack_provider = get_callstack_provider(args, context)
    dispatcher_data = get_dispatcher_data(args, context)

    callstacks_data = run_checks.run_per_core_check(
        lambda risc_debug: dump_lightweight_asserts(
            risc_debug,
            dispatcher_data,
            callstack_provider,
        ),
        block_filter=BLOCK_TYPES_TO_CHECK,
    )

    return callstacks_data


if __name__ == "__main__":
    run_script()
