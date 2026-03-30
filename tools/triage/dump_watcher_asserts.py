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


# FNV-1a 16-bit folded hash — matches device-side constexpr debug_file_hash in dev_msgs.h
def _debug_file_hash(s: str) -> int:
    h = 2166136261
    for c in s:
        h = ((h ^ ord(c)) * 16777619) & 0xFFFFFFFF
    return ((h >> 16) ^ (h & 0xFFFF)) & 0xFFFF


def _resolve_file_from_hash_via_elf(file_id: int, parsed_elf) -> str | None:
    """
    Resolve a file_id hash to a source path by scanning the file paths recorded
    in the ELF's DWARF line table — the same paths the compiler stored for __FILE__.
    Strips the CWD prefix to recover the relative path used at compile time.
    """
    repo_root = os.getcwd()
    if not repo_root.endswith("/"):
        repo_root += "/"

    seen: set[str] = set()
    for (_start, _end), (dwarf_fname, _line, _col) in parsed_elf._dwarf.file_lines_ranges.items():
        if dwarf_fname in seen:
            continue
        seen.add(dwarf_fname)
        # Try the path as stored in DWARF (may already be relative).
        if _debug_file_hash(dwarf_fname) == file_id:
            return dwarf_fname
        # Strip repo-root prefix to get the relative path the compiler saw via __FILE__.
        if dwarf_fname.startswith(repo_root):
            rel = dwarf_fname[len(repo_root) :]
            if _debug_file_hash(rel) == file_id:
                return rel
    return None


def _find_assert_address_in_elf(parsed_elf, filename: str, line_num: int) -> int | None:
    """Scan the DWARF line table for the first address at (filename, line_num)."""
    for (start, _end), (dwarf_fname, line, _col) in parsed_elf._dwarf.file_lines_ranges.items():
        if line == line_num and (dwarf_fname == filename or dwarf_fname.endswith(filename)):
            return start
    return None


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
    file: str = triage_field("File")
    line: int = triage_field("Line")
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
        file_id = int(mailboxes.watcher.assert_status.file_id)
        line_num = int(mailboxes.watcher.assert_status.line_num)
    except Exception:
        return None

    # Resolve the source file by scanning DWARF paths in the ELF — the same paths
    # the compiler stored for __FILE__, which is what debug_file_hash hashed.
    resolved_file: str | None = None
    for elf_path in [dispatcher_core_data.kernel_path, dispatcher_core_data.firmware_path]:
        if not elf_path:
            continue
        try:
            parsed_elf = elfs_cache[elf_path]
            resolved_file = _resolve_file_from_hash_via_elf(file_id, parsed_elf)
            if resolved_file:
                break
        except Exception:
            continue

    file_str = resolved_file if resolved_file else f"unknown file (hash=0x{file_id:04X})"

    # Look up the assert address in DWARF to get a proper callstack frame.
    callstack_with_msg: KernelCallstackWithMessage | None = None
    if resolved_file:
        context = location._device._context
        for elf_path, offset in [
            (dispatcher_core_data.kernel_path, dispatcher_core_data.kernel_offset),
            (dispatcher_core_data.firmware_path, None),
        ]:
            if not elf_path:
                continue
            try:
                parsed_elf = elfs_cache[elf_path]
                addr = _find_assert_address_in_elf(parsed_elf, resolved_file, line_num)
                if addr is None:
                    continue
                cs = top_callstack(addr, parsed_elf, offset, context)
                if cs:
                    callstack_with_msg = KernelCallstackWithMessage(callstack=cs, message=None)
                    break
            except Exception:
                continue

    if callstack_with_msg is None:
        callstack_with_msg = KernelCallstackWithMessage(
            callstack=[CallstackEntry(pc=None, function_name=None, file=resolved_file, line=line_num)],
            message="Rebuild with TT_METAL_RISCV_DEBUG_INFO=1 + TT_METAL_WATCHER=1 for full callstack",
        )

    # Read arguments/locals from the full PC-based callstack — frame matching resolved_file.
    parameters: str | None = None
    try:
        pc_data = callstack_provider.get_cached_callstacks(location, risc_name, use_full_callstack=True)
        for frame in pc_data.kernel_callstack_with_message.callstack:
            if frame.file and resolved_file and frame.file.endswith(resolved_file):
                all_vars = list(frame.arguments) + list(frame.locals)
                if all_vars:
                    parameters = _serialize_parameters(all_vars)
                break
    except Exception:
        pass

    return WatcherAssertResult(
        kernel_name=dispatcher_core_data.kernel_name,
        file=file_str,
        line=line_num,
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
