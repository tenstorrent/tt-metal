#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Usage:
    check_go_message

Description:
    Records and validates the go-message handshake state on Tensix cores that were mid-flight when
    a hang was captured. None of this state is preserved by the other triage scripts today.

    Kernels are preloaded (DISPATCH_ENABLE_FLAG_PRELOAD is set unconditionally in
    tt_metal/impl/program/dispatch.cpp), so brisc launches its subordinates before the go signal
    arrives and each subordinate then parks in wait_for_go_message()
    (tt_metal/hw/inc/internal/firmware_common.h), polling
    mailboxes->go_messages[mailboxes->go_message_index].signal until it reads RUN_MSG_GO. brisc only
    writes RUN_MSG_DONE after wait_ncrisc_trisc() returns, so while a subordinate is still stuck at
    that barrier the slot it should be polling still reads GO.

    A hang where some subordinates on a core cleared the barrier and others did not therefore leaves
    two candidate explanations, and this script exists to record the evidence that separates them:

      - The handshake state itself is wrong (for example go_message_index is out of range, which no
        firmware path bounds-checks). Reported here as a failure.
      - The handshake state is correct and a subordinate simply failed to observe it. Reported here
        as a warning carrying the full mailbox state, which is the forensic record.

    Limitation: a subordinate's *latched* copy of go_message_index lives in a RISC register, not in
    L1, so it cannot be read back from the mailbox. This script constrains the shared state only.
    To exclude a stale latch entirely, wait_for_go_message() must re-read the index per iteration
    (as brisc.cc and the tt-2xx dm firmware already do).

Owner:
    jamesleeTT
"""

from ttexalens.coordinate import OnChipCoordinate
from ttexalens.context import Context

from dispatcher_data import run as get_dispatcher_data, DispatcherData
from run_checks import run as get_run_checks
from triage import ScriptConfig, log_check_location, log_warning_location, run_script

script_config = ScriptConfig(
    depends=["run_checks", "dispatcher_data"],
)


def format_slots(signals: list[int], state_names: dict[int, str], index: int) -> str:
    """Render every go_messages[] entry, marking the one the core is polling with '*'."""
    return " ".join(
        f"[{slot}]{'*' if slot == index else ''}={state_names.get(signal, str(signal))}"
        for slot, signal in enumerate(signals)
    )


def check_go_message(
    location: OnChipCoordinate,
    risc_name: str,
    dispatcher_data: DispatcherData,
    num_entries: int,
    state_names: dict[int, str],
):
    core_data = dispatcher_data.get_cached_core_data(location, risc_name)

    # risc_enabled_by_kernel is None when the read failed or the mailbox is corrupt; only skip cores
    # we positively know had no kernel launched on them.
    if core_data.risc_enabled_by_kernel is False:
        return

    mailboxes = core_data.mailboxes
    if mailboxes is None:
        return

    # Read the index on its own, so an unreadable slot below does not also cost us the index value.
    try:
        index = int(mailboxes.go_message_index)
    except Exception:
        log_check_location(location, False, f"{risc_name}: could not read go_message_index")
        return

    signals: list[int] = []
    for slot in range(num_entries):
        try:
            signals.append(int(mailboxes.go_messages[slot].signal))
        except Exception:
            log_check_location(
                location,
                False,
                f"{risc_name}: could not read go_messages[{slot}].signal (go_message_index={index})",
            )
            return

    if not 0 <= index < num_entries:
        log_check_location(
            location,
            False,
            f"{risc_name}: go_message_index={index} is outside [0, {num_entries}). "
            f"wait_for_go_message() does not bounds-check it, so a subordinate that read this value "
            f"polls unmapped mailbox memory and never observes GO. "
            f"Slots: {format_slots(signals, state_names, index)}",
        )
        return

    # A core whose slot reads DONE finished normally. Everything else was mid-flight when the hang
    # was captured, which is exactly the state worth preserving. Unused slots are documented as
    # potentially holding garbage (dispatch.cpp), so their contents are reported, not asserted on.
    if core_data.go_message == "DONE":
        return

    log_warning_location(
        location,
        f"{risc_name}: mid-flight at hang time — go_message_index={index}, "
        f"waypoint={core_data.waypoint}, subordinate_sync={core_data.subordinate_sync}, "
        f"slots: {format_slots(signals, state_names, index)}",
    )


def run(args, context: Context):
    BLOCK_TYPES_TO_CHECK = ["tensix"]
    # The go message and its index are core-wide, so inspect one RISC per core.
    RISC_CORES_TO_CHECK = ["brisc"]

    dispatcher_data = get_dispatcher_data(args, context)
    run_checks = get_run_checks(args, context)

    num_entries = dispatcher_data._brisc_elf.get_constant("go_message_num_entries")
    assert isinstance(num_entries, int)
    state_names = dispatcher_data._go_message_states

    run_checks.run_per_core_check(
        lambda location, risc_name: check_go_message(location, risc_name, dispatcher_data, num_entries, state_names),
        block_filter=BLOCK_TYPES_TO_CHECK,
        core_filter=RISC_CORES_TO_CHECK,
    )


if __name__ == "__main__":
    run_script()
