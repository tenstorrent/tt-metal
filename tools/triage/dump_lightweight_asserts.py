#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
from dataclasses import dataclass
from triage import ScriptConfig, log_check_risc, run_script, triage_field
from callstack_provider import (
    KernelCallstackWithMessage,
    format_callstack_with_message,
    run as get_callstack_provider,
    CallstackProvider,
    get_function_die,
    extract_template_params,
)
from run_checks import run as get_run_checks
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.context import Context
from ttexalens.tt_exalens_lib import read_word_from_device
from ttexalens.hardware.risc_debug import CallstackEntryVariable
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
            while True:
                new_index = code_line.find("ASSERT(", start_index + 1)
                if new_index == -1:
                    break
                # Walk backward to find the actual start of the macro name
                # (e.g. "LLK_ASSERT(" -> position of 'L', not 'A')
                macro_start = new_index
                while macro_start > 0 and (code_line[macro_start - 1].isalnum() or code_line[macro_start - 1] == "_"):
                    macro_start -= 1
                if column is not None and macro_start >= column:
                    break
                start_index = new_index
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


def serialize_variables(variables: list[CallstackEntryVariable], assert_code: str) -> str:
    result = ""
    for var in variables:
        var_name = var.name or "?"
        var_value = var.value if var.value is not None else "?"
        if var_name in assert_code:
            serialized = f"- [info]{var_name}[/] = [command]{var_value}[/]\n"
        else:
            serialized = f"- {var_name} = {var_value}\n"
        result += serialized
    return result


def dump_lightweight_asserts(
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
            dispatcher_core_data = callstack_provider.dispatcher_data.get_cached_core_data(location, risc_name)
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

        # Check if core hit ebreak or is spinning in a while(true) loop after an assert failure
        while_true_instruction = 0x0000006F
        ebreak_instruction = 0x00100073
        rewind_pc_for_ebreak = False
        if previous_instruction == ebreak_instruction:
            rewind_pc_for_ebreak = True
        elif current_instruction != ebreak_instruction and current_instruction != while_true_instruction:
            return None

        callstack_data = callstack_provider.get_cached_callstacks(
            location, risc_name, rewind_pc_for_ebreak, use_full_callstack=True
        )
        arguments_and_locals = None
        assert_code = "?"
        if callstack_data.kernel_callstack_with_message.callstack[0] is not None:
            assert_code = extract_assert_code(
                callstack_data.kernel_callstack_with_message.callstack[0].file,
                callstack_data.kernel_callstack_with_message.callstack[0].line,
                callstack_data.kernel_callstack_with_message.callstack[0].column,
            )
            arguments_and_locals = ""
            top_frame = callstack_data.kernel_callstack_with_message.callstack[0]
            try:
                function_die = get_function_die(top_frame)
                if function_die is not None:
                    template_params = extract_template_params(function_die)
                    if template_params:
                        arguments_and_locals += "\nTemplate arguments:\n"
                        for param_name, param_value in template_params:
                            if param_name and param_name in assert_code:
                                arguments_and_locals += f"- [info]{param_name}[/] = [command]{param_value}[/]\n"
                            else:
                                arguments_and_locals += f"- {param_name or '?'} = {param_value}\n"
                        for param_name, _ in template_params:
                            if param_name:
                                assert_code = assert_code.replace(param_name, f"[info]{param_name}[/]")
            except Exception as e:
                if os.getenv("TT_TRIAGE_DEBUG_TEMPLATES"):
                    log_check_risc(
                        risc_name,
                        location,
                        False,
                        f"[warning]Template arguments unavailable: {e}[/]",
                    )
            if len(top_frame.arguments) > 0:
                arguments_and_locals += "\nRuntime arguments:\n"
                arguments_and_locals += serialize_variables(top_frame.arguments, assert_code)
                for var in top_frame.arguments:
                    if var.name is not None:
                        assert_code = assert_code.replace(var.name, f"[info]{var.name}[/]")
            if len(top_frame.locals) > 0:
                arguments_and_locals += "\nLocals:\n"
                arguments_and_locals += serialize_variables(top_frame.locals, assert_code)
                for var in top_frame.locals:
                    if var.name is not None:
                        assert_code = assert_code.replace(var.name, f"[info]{var.name}[/]")
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
    BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth", "active_eth"]

    run_checks = get_run_checks(args, context)
    callstack_provider = get_callstack_provider(args, context)

    callstacks_data = run_checks.run_per_core_check(
        lambda location, risc_name: dump_lightweight_asserts(
            location,
            risc_name,
            callstack_provider,
        ),
        block_filter=BLOCK_TYPES_TO_CHECK,
    )

    return callstacks_data


if __name__ == "__main__":
    run_script()
