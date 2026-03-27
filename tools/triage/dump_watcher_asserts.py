#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_watcher_asserts

Description:
    Dumps watcher ASSERT() and ASSERT_MSG() failures recorded in the device mailbox.

    When TT_METAL_WATCHER=1 is active and a kernel trips an ASSERT() or ASSERT_MSG(),
    the device stores the following in mailboxes.watcher.assert_status:
      - file_id  : FNV-1a hash of __FILE__ (identifies the source file)
      - line_num : line number where the assert macro appears
      - tripped  : assert type enum (DebugAssertTripped = 3)
      - which    : RISC processor index that tripped the assert

    This script resolves file_id back to a source path, extracts the ASSERT_MSG message
    string directly from that file at line_num (handling multi-line macros), and looks up
    the assert site in the kernel ELF DWARF to produce a proper callstack frame.

Owner:
    dstoiljkovic-TT
"""

import os
from dataclasses import dataclass

from triage import ScriptConfig, run_script, triage_field
from callstack_provider import (
    KernelCallstackWithMessage,
    format_callstack_with_message,
    run as get_callstack_provider,
    CallstackProvider,
)
from dispatcher_data import DispatcherData
from elfs_cache import ElfsCache
from run_checks import run as get_run_checks
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.context import Context
from ttexalens.hardware.risc_debug import CallstackEntry, CallstackEntryVariable
from ttexalens.tt_exalens_lib import top_callstack


script_config = ScriptConfig(
    depends=["run_checks", "callstack_provider"],
)

# Matches enum debug_assert_type_t in tt_metal/hw/inc/hostdev/dev_msgs.h
_WATCHER_ASSERT_TRIPPED = 3


# ── Per-core result dataclass ─────────────────────────────────────────────────


def _fmt_callstack(cs_with_msg: KernelCallstackWithMessage) -> str:
    return format_callstack_with_message(cs_with_msg)


def _serialize_parameters(variables: list[CallstackEntryVariable]) -> str:
    lines = []
    for var in variables:
        name = var.name or "?"
        value = var.value if var.value is not None else "?"
        lines.append(f"  {name} = {value}")
    return "\n".join(lines) if lines else "(none)"


@dataclass
class WatcherAssertResult:
    kernel_name: str | None = triage_field("Kernel Name")
    pc: str = triage_field("PC")
    parameters: str | None = triage_field("Parameters")
    callstack: KernelCallstackWithMessage = triage_field("Callstack", _fmt_callstack)


# ── Core check ────────────────────────────────────────────────────────────────


def check_watcher_assert(
    location: OnChipCoordinate,
    risc_name: str,
    callstack_provider: CallstackProvider,
) -> WatcherAssertResult | None:
    dispatcher_data: DispatcherData = callstack_provider.dispatcher_data
    elfs_cache: ElfsCache = callstack_provider.elfs_cache

    dispatcher_core_data = dispatcher_data.get_cached_core_data(location, risc_name)
    mailboxes = dispatcher_core_data.mailboxes
    if mailboxes is None:
        return None

    try:
        tripped = int(mailboxes.watcher.assert_status.tripped)
    except Exception:
        return None

    if tripped != _WATCHER_ASSERT_TRIPPED:
        return None

    try:
        # pc is the return address captured by __builtin_return_address(0) inside assert_and_hang,
        # pointing to the call site inside the ASSERT macro expansion.
        pc = int(mailboxes.watcher.assert_status.pc)
    except Exception:
        return None

    # Use top_callstack exactly as triage does: DWARF resolves pc → file, line, function, message.
    context = location._device._context
    callstack_with_msg: KernelCallstackWithMessage | None = None

    for elf_path, offset in [
        (dispatcher_core_data.kernel_path, dispatcher_core_data.kernel_offset),
        (dispatcher_core_data.firmware_path, None),
    ]:
        if not elf_path:
            continue
        try:
            parsed_elf = elfs_cache[elf_path]
            cs = top_callstack(pc, parsed_elf, offset, context)
            if cs:
                callstack_with_msg = KernelCallstackWithMessage(callstack=cs, message=None)
                break
        except Exception:
            continue

    if callstack_with_msg is None:
        callstack_with_msg = KernelCallstackWithMessage(
            callstack=[CallstackEntry(pc=pc, function_name=None, file=None, line=None)],
            message="Rebuild with TT_METAL_RISCV_DEBUG_INFO=1 for full resolution",
        )

    # Read arguments/locals from the assert frame (frame #0 = return site inside ASSERT macro).
    parameters: str | None = None
    try:
        if callstack_with_msg.callstack:
            frame = callstack_with_msg.callstack[0]
            all_vars: list[CallstackEntryVariable] = list(frame.arguments) + list(frame.locals)
            if all_vars:
                parameters = _serialize_parameters(all_vars)
    except Exception:
        pass

    return WatcherAssertResult(
        kernel_name=dispatcher_core_data.kernel_name,
        pc=f"0x{pc:08X}",
        parameters=parameters,
        callstack=callstack_with_msg,
    )


# ── Script entry point ────────────────────────────────────────────────────────


def run(args, context: Context):
    BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth", "active_eth"]

    run_checks = get_run_checks(args, context)
    callstack_provider = get_callstack_provider(args, context)

    return run_checks.run_per_core_check(
        lambda location, risc_name: check_watcher_assert(location, risc_name, callstack_provider),
        block_filter=BLOCK_TYPES_TO_CHECK,
    )


if __name__ == "__main__":
    run_script()
