#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    dump_callstacks [--all-cores]

Options:
    --all-cores        Show all cores including ones with Go Message = DONE. By default, DONE cores are filtered out.

Description:
    Dumps callstacks for all devices in the system and for every supported risc processor.
    By default, filters out cores with DONE status and shows essential fields.
    Use --all-cores to see all cores, and -v/-vv to show more columns.

    Color output is automatically enabled when stdout is a TTY (terminal) and can be overridden
    with TT_TRIAGE_COLOR environment variable (0=disable, 1=enable).
"""

from triage import ScriptConfig, log_check_risc, run_script
from callstack_provider import run as get_callstack_provider, CallstackProvider, CallstacksData
from run_checks import run as get_run_checks
from ttexalens.coordinate import OnChipCoordinate
from ttexalens.context import Context
from utils import ORANGE, RST

script_config = ScriptConfig(
    depends=["run_checks", "callstack_provider"],
)


def dump_callstacks(
    location: OnChipCoordinate,
    risc_name: str,
    callstack_provider: CallstackProvider,
    show_all_cores: bool = False,
) -> CallstacksData | None:
    try:
        # Skip DONE cores unless --all-cores is specified
        if not show_all_cores:
            dispatcher_core_data = callstack_provider.dispatcher_data.get_cached_core_data(location, risc_name)
            if dispatcher_core_data.go_message == "DONE":
                return None
        return callstack_provider.get_callstacks(location, risc_name)
    except Exception as e:
        log_check_risc(
            risc_name,
            location,
            False,
            f"{ORANGE}Failed to dump callstacks: {e}{RST}",
        )
        return None


def run(args, context: Context):
    from triage import set_verbose_level

    show_all_cores: bool = args["--all-cores"]

    BLOCK_TYPES_TO_CHECK = ["tensix", "idle_eth", "active_eth"]

    run_checks = get_run_checks(args, context)
    callstack_provider = get_callstack_provider(args, context)

    return run_checks.run_per_core_check(
        lambda location, risc_name: dump_callstacks(
            location,
            risc_name,
            callstack_provider,
            show_all_cores,
        ),
        block_filter=BLOCK_TYPES_TO_CHECK,
    )


if __name__ == "__main__":
    run_script()
