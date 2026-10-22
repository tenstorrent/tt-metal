#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_callstacks [--all-cores]

Options:
    --all-cores        Show all cores, including those with Go Message = DONE and RISCs not enabled
                       by the running program. By default, both are filtered out.

Description:
    Dumps callstacks for all devices in the system and for every supported RISC processor.
    By default, shows essential fields and filters out cores with Go Message = DONE and RISCs not
    enabled by the running program (kernel_config.enables).
    Use --all-cores to see all cores, and -v/-vv to show more columns.
    With --full-callstack, local variable values shown as "?" could not be read (e.g. optimized out).

    Color output is automatically enabled when stdout is a TTY (terminal) and can be overridden
    with TT_TRIAGE_COLOR environment variable (0=disable, 1=enable).

Owner:
    tt-vjovanovic
"""

import os
from dataclasses import dataclass

from triage import ScriptConfig, log_check_risc, run_script, triage_field
from callstack_provider import (
    run as get_callstack_provider,
    CallstackProvider,
    CallstacksData,
    serialize_variables,
)
from run_checks import run as get_run_checks
from ttexalens.hardware.risc_debug import RiscDebug
from ttexalens.context import Context
from ttexalens.umd_device import TimeoutDeviceRegisterError
from ttexalens.elf import CallstackEntry

script_config = ScriptConfig(
    depends=["run_checks", "callstack_provider"],
    disabled=os.environ.get("TT_TRIAGE_ENABLE_AGGREGATED_CALLSTACKS") == "1",
)


@dataclass
class CallstacksDataWithLocals(CallstacksData):
    """CallstacksData plus the Locals column, used only with --full-callstack."""

    frame_locals: str | None = triage_field("Locals")


def _format_frame_locals(callstack: list[CallstackEntry]) -> str | None:
    """Text for the Locals column: locals grouped under their frame number, or None if no local has a value."""
    lines = []
    for i, frame in enumerate(callstack):
        if all(var.value is None for var in frame.locals):
            continue
        prefix = f"#{i} "
        for line in serialize_variables(frame.locals).splitlines():
            lines.append(prefix + line)
            prefix = " " * len(prefix)
    return "\n".join([""] + lines) if lines else None


def dump_callstacks(
    risc_debug: RiscDebug,
    callstack_provider: CallstackProvider,
    show_all_cores: bool = False,
) -> CallstacksData | None:
    risc_location = risc_debug.risc_location
    location = risc_location.location
    risc_name = risc_location.risc_name
    try:
        if not callstack_provider.dispatcher_data.risc_enabled(risc_name):
            return None
        # Skip DONE / not-enabled-by-design cores unless --all-cores is specified
        if not show_all_cores and callstack_provider.dispatcher_data.is_idle_in_default_view(risc_location):
            return None
        callstacks = callstack_provider.get_cached_callstacks(risc_debug)
        if not callstack_provider.full_callstack or callstack_provider.gdb_callstack:
            return callstacks
        frame_locals = _format_frame_locals(callstacks.kernel_callstack_with_message.callstack)
        return CallstacksDataWithLocals(**vars(callstacks), frame_locals=frame_locals)
    except TimeoutDeviceRegisterError:
        raise
    except Exception as e:
        log_check_risc(
            risc_name,
            location,
            False,
            f"[warning]Failed to dump callstacks: {e}[/]",
        )
        return None


def run(args, context: Context):
    show_all_cores: bool = args["--all-cores"]
    BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth", "active_eth", "dram"]
    run_checks = get_run_checks(args, context)
    callstack_provider = get_callstack_provider(args, context)
    return run_checks.run_per_core_check(
        lambda risc_debug: dump_callstacks(
            risc_debug,
            callstack_provider,
            show_all_cores,
        ),
        block_filter=BLOCK_TYPES_TO_CHECK,
    )


if __name__ == "__main__":
    run_script()
