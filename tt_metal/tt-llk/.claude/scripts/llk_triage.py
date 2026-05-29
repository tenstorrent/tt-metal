#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LLK-specific device triage.

Used by `.claude/scripts/run_test.sh` on a silicon hang. Also runnable
standalone:

    python3 .claude/scripts/llk_triage.py --arch blackhole \
        [--location 0,0] [--device-id 0]

Why a separate triage from tt-metal's tt-triage:
tt-metal/tools/tt-triage.py reads device state through Metal's Inspector
subsystem (live RPC or `/tmp/tt-metal/inspector` log dir). LLK tests run
directly on the device through ttexalens; no Metal process is in the
loop, so Inspector data doesn't exist. tt-triage's checks then all fail
their dependency on `inspector_data` and print
"Cannot run script due to failed dependencies".

This script bypasses Inspector. It reads three signal layers directly
via ttexalens (the same channel the LLK test harness uses):

  1. Mailbox state — per-thread completion sentinels in L1.
  2. Per-RISC debug state — `get_pc()`, `is_in_reset()`,
     `is_halted()`, `is_ebreak_hit()` per BRISC/TRISC0..3. PC is
     sampled twice with a short delay so "advancing" vs "stuck"
     can be reported.
  3. Tensix CFG registers — `get_tensix_state()` dump (ALU, unpack,
     pack, relu config groups + GPRs + debug-bus counters).

Output is plain text suitable for inclusion in the `RUN_LLK_TESTS_HANG`
block.

Mailbox semantics (per kernel thread):
  0xA3  host reset sentinel — `reset_mailboxes` writes this before the
        test. The launch sequence (in both BRISC-boot and TRISC-boot
        modes) writes RESET_VAL to all kernel mailboxes BEFORE calling
        `clear_trisc_soft_reset()` to release the TRISCs. So a mailbox
        still reading 0xA3 implies the writer (BRISC firmware in
        BRISC-boot mode, or the Unpack TRISC in TRISC-boot mode) never
        executed that block — and therefore the TRISC for this mailbox
        is still in soft-reset and cannot be running kernel code. The
        failure is in the launch path, not the kernel itself.
  0x00  RESET_VAL — written at kernel entry by BRISC (BRISC boot mode)
        or by the Unpack TRISC (TRISC boot mode). The kernel passed the
        entry point. It may still be running, or it may have wedged
        partway. The mailbox alone CANNOT distinguish these two; the
        per-RISC PC sampled below is what tells you whether the core is
        still advancing.
  0xFF  KERNEL_COMPLETE — the final store in `main()` after the kernel
        body returns. The thread finished.
  *     any other value indicates corruption.

To classify a 0x00 thread as "still running" vs "stuck", the triage
samples the per-RISC PC twice with a short delay and combines it with
the `is_in_reset` / `is_halted` / `is_ebreak_hit` flags read directly
from `risc_debug`. That is what discriminates RESET_VAL-just-written
from kernel-actively-cooking from kernel-hit-an-assert from kernel-
spinning-on-a-condition.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

KERNEL_COMPLETE = 0xFF
KERNEL_STARTED = 0x00  # RESET_VAL written by BRISC/TRISC at kernel entry
HOST_RESET = 0xA3  # written by reset_mailboxes() before each test
KERNEL_THREAD_MAILBOXES = {"Unpacker", "Math", "Packer", "Sfpu"}

# Maps each kernel-thread mailbox name to the RISC core that runs it.
# Used to look up the right PC field when classifying "running vs stuck".
MAILBOX_TO_RISC = {
    "Unpacker": "TRISC0",
    "Math": "TRISC1",
    "Packer": "TRISC2",
    "Sfpu": "TRISC3",  # Quasar only
}


def _classify_mailbox(val: int) -> str:
    """Return the short tag for a mailbox value."""
    if val == KERNEL_COMPLETE:
        return "KERNEL_COMPLETE"
    if val == KERNEL_STARTED:
        return "KERNEL_STARTED"  # passed RESET_VAL write; could be running OR stuck
    if val == HOST_RESET:
        return "HOST_RESET"
    return "UNKNOWN"


def _add_llk_helpers_to_path() -> None:
    """Put tests/python_tests/ on sys.path so we can import Mailboxes enums."""
    here = os.path.dirname(os.path.abspath(__file__))
    worktree = os.path.abspath(os.path.join(here, "..", ".."))
    helpers_root = os.path.join(worktree, "tests", "python_tests")
    if helpers_root not in sys.path:
        sys.path.insert(0, helpers_root)


def _risc_names_for_arch(arch: str) -> list[str]:
    """Quasar has 4 TRISCs and no BRISC; WH/BH have BRISC + 3 TRISCs."""
    if arch == "quasar":
        return ["TRISC0", "TRISC1", "TRISC2", "TRISC3"]
    return ["BRISC", "TRISC0", "TRISC1", "TRISC2"]


def _get_risc_debug(block, risc_name: str, arch: str):
    """Wrap `block.get_risc_debug(...)` with the arch-specific neo_id kwarg.

    Quasar requires neo_id=0; passing it on WH/BH raises.
    """
    if arch == "quasar":
        return block.get_risc_debug(risc_name, neo_id=0)
    return block.get_risc_debug(risc_name)


def _snapshot_risc_state(risc_debug) -> dict:
    """Read PC + status flags for one RISC. Individual fields default to None
    if the underlying call raises (e.g. core in reset can't be queried)."""
    snap: dict = {
        "pc": None,
        "is_in_reset": None,
        "is_halted": None,
        "is_ebreak_hit": None,
        "errors": [],
    }
    try:
        snap["is_in_reset"] = risc_debug.is_in_reset()
    except Exception as exc:
        snap["errors"].append(f"is_in_reset: {exc}")
    # If the core is in reset, the rest of the queries either fail loudly or
    # return junk — skip them and let the consumer see is_in_reset=True.
    if snap["is_in_reset"]:
        return snap
    for field, call in (
        ("is_halted", risc_debug.is_halted),
        ("is_ebreak_hit", risc_debug.is_ebreak_hit),
        ("pc", risc_debug.get_pc),
    ):
        try:
            snap[field] = call()
        except Exception as exc:
            snap["errors"].append(f"{field}: {exc}")
    return snap


def _collect_risc_states(
    arch: str, location: str, device_id: int, check_context, convert_coordinate
) -> dict[str, dict]:
    """Build {risc_name: snapshot} sampled twice with a short delay.

    Each snapshot has pc/pc2, is_in_reset, is_halted, is_ebreak_hit. The two
    PC samples support the "advancing vs stuck" classifier without changing
    the user-visible record shape.
    """
    context = check_context()
    coordinate = convert_coordinate(location, device_id, context)
    device = context.devices[device_id]
    block = device.get_block(coordinate)

    debug_handles: dict[str, object] = {}
    for risc_name in _risc_names_for_arch(arch):
        try:
            debug_handles[risc_name] = _get_risc_debug(block, risc_name, arch)
        except Exception as exc:
            debug_handles[risc_name] = exc

    states: dict[str, dict] = {}
    for risc_name, handle in debug_handles.items():
        if isinstance(handle, Exception):
            states[risc_name] = {"errors": [f"get_risc_debug: {handle}"]}
            continue
        states[risc_name] = _snapshot_risc_state(handle)

    # Second PC sample after a short gap. Only re-read PC for cores that
    # weren't in reset on the first pass — anything in reset can't advance.
    time.sleep(0.05)
    for risc_name, handle in debug_handles.items():
        if isinstance(handle, Exception):
            continue
        snap = states[risc_name]
        if snap.get("is_in_reset"):
            snap["pc2"] = None
            continue
        try:
            snap["pc2"] = handle.get_pc()
        except Exception as exc:
            snap["pc2"] = None
            snap["errors"].append(f"pc2: {exc}")
    return states


def _classify_started_threads(
    started: list[str], risc_states: dict[str, dict]
) -> dict[str, str]:
    """For each `KERNEL_STARTED` thread, combine status flags + PC delta into
    a one-line verdict.

    Returns {mailbox_name: "<verdict>"}.
    """
    verdicts: dict[str, str] = {}
    for mb_name in started:
        risc = MAILBOX_TO_RISC.get(mb_name)
        snap = risc_states.get(risc) if risc else None
        if snap is None:
            verdicts[mb_name] = "unknown (no risc_debug)"
            continue
        if snap.get("is_in_reset"):
            # Mailbox says we passed entry but RISC is now in reset: launch
            # path raced, or someone re-asserted reset after the kernel began.
            verdicts[mb_name] = "in soft-reset (post-entry)"
            continue
        if snap.get("is_ebreak_hit"):
            pc = snap.get("pc")
            pc_str = f"0x{pc:x}" if isinstance(pc, int) else "?"
            verdicts[mb_name] = f"asserted (ebreak @ {pc_str})"
            continue
        if snap.get("is_halted"):
            pc = snap.get("pc")
            pc_str = f"0x{pc:x}" if isinstance(pc, int) else "?"
            verdicts[mb_name] = f"halted (non-ebreak) @ {pc_str}"
            continue
        pc1, pc2 = snap.get("pc"), snap.get("pc2")
        if not isinstance(pc1, int) or not isinstance(pc2, int):
            verdicts[mb_name] = "unknown (pc unreadable)"
        elif pc1 != pc2:
            verdicts[mb_name] = f"advancing (pc {hex(pc1)} → {hex(pc2)})"
        else:
            verdicts[mb_name] = f"stuck @ 0x{pc1:x}"
    return verdicts


def _print_risc_states(risc_states: dict[str, dict]) -> None:
    """Print the per-RISC debug snapshot as its own section so the master
    agent sees PC / reset / halt / ebreak fields verbatim, not just the
    mailbox-derived verdicts."""
    print("== RISC debug state ==")
    if not risc_states:
        print("  (no risc_debug data collected)")
        return
    for risc_name, snap in risc_states.items():
        if "is_halted" not in snap and "is_in_reset" not in snap:
            err = "; ".join(snap.get("errors", [])) or "unavailable"
            print(f"  {risc_name:<7}  {err}")
            continue
        in_reset = snap.get("is_in_reset")
        halted = snap.get("is_halted")
        ebreak = snap.get("is_ebreak_hit")
        pc = snap.get("pc")
        pc2 = snap.get("pc2")
        pc_str = f"0x{pc:x}" if isinstance(pc, int) else "—"
        pc2_str = f"0x{pc2:x}" if isinstance(pc2, int) else "—"
        delta = ""
        if isinstance(pc, int) and isinstance(pc2, int):
            delta = " (advancing)" if pc != pc2 else " (no change)"
        print(
            f"  {risc_name:<7}  in_reset={in_reset}  halted={halted}  "
            f"ebreak={ebreak}  pc={pc_str}  pc2={pc2_str}{delta}"
        )
        for err in snap.get("errors", []):
            print(f"             ! {err}")


def _print_mailbox_state(
    mailbox_cls, location: str, device_id: int, read_word, risc_states: dict[str, dict]
) -> list[str]:
    print("== Mailbox state ==")
    started: list[str] = []
    never_started: list[str] = []
    unknown: list[str] = []
    for mb in mailbox_cls:
        try:
            val = read_word(location, mb.value, device_id=device_id)
            if mb.name in KERNEL_THREAD_MAILBOXES:
                # Kernel-thread mailbox: classify against the KERNEL_COMPLETE /
                # RESET_VAL / HOST_RESET sentinels.
                tag = _classify_mailbox(val)
                print(f"  {mb.name:<14} @ 0x{mb.value:08X}  =  0x{val:08X}  ({tag})")
                if tag == "KERNEL_STARTED":
                    started.append(mb.name)
                elif tag == "HOST_RESET":
                    never_started.append(mb.name)
                elif tag == "UNKNOWN":
                    unknown.append(mb.name)
            else:
                # Host↔BRISC protocol slot (command buffer, counter, bread). These
                # hold protocol values, not kernel-completion sentinels — printing
                # them is useful for diagnosing the launch path, but classifying
                # them against KERNEL_COMPLETE would be misleading.
                print(
                    f"  {mb.name:<14} @ 0x{mb.value:08X}  =  0x{val:08X}  (BRISC protocol)"
                )
        except Exception as exc:
            print(f"  {mb.name:<14} @ 0x{mb.value:08X}  =  READ_ERROR: {exc}")
    print()
    hung_threads = started + never_started + unknown
    if started:
        verdict = _classify_started_threads(started, risc_states)
        labeled = [f"{name} ({verdict.get(name, 'unknown')})" for name in started]
        print(f"Kernel threads past entry, not yet complete: {', '.join(labeled)}")
    if never_started:
        print(
            f"Kernel threads never started (host sentinel intact): {', '.join(never_started)}"
        )
    if unknown:
        print(f"Kernel threads with unexpected mailbox value: {', '.join(unknown)}")
    if not hung_threads:
        print("Hung kernel threads: (none — all kernel mailboxes hit KERNEL_COMPLETE)")
    return hung_threads


def _print_tensix_cfg(location: str, device_id: int, get_tensix_state) -> None:
    """Dump Tensix CFG-register groups (ALU/unpack/pack/relu config, GPRs, debug-bus
    counters). This is the `TensixState` dataclass from `tt_exalens_lib`."""
    print("== Tensix CFG registers ==")
    try:
        state = get_tensix_state(location, device_id=device_id)
        print(json.dumps(asdict(state), indent=2, default=str))
    except Exception as exc:
        print(f"  get_tensix_state failed: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLK device triage via ttexalens (no Metal/Inspector required).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--arch",
        choices=["blackhole", "wormhole", "quasar"],
        required=True,
        help="Target architecture (selects the right Mailboxes enum)",
    )
    parser.add_argument(
        "--location",
        default="0,0",
        help="Tensix logical location 'row,col' (default: 0,0 — matches TestConfig)",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Device id (default: 0)",
    )
    args = parser.parse_args()

    _add_llk_helpers_to_path()

    try:
        from ttexalens.tt_exalens_lib import (
            check_context,
            convert_coordinate,
            get_tensix_state,
            read_word_from_device,
        )
    except ImportError as exc:
        print(
            f"[llk-triage] ttexalens not importable in this Python: {exc}",
            file=sys.stderr,
        )
        print(
            "[llk-triage] activate tests/.venv before running, or invoke via run_test.sh",
            file=sys.stderr,
        )
        return 2

    try:
        from helpers.llk_params import (  # type: ignore[import-not-found]
            Mailboxes,
            MailboxesQuasar,
        )
    except ImportError as exc:
        print(f"[llk-triage] LLK helpers not importable: {exc}", file=sys.stderr)
        return 2

    mailbox_cls = MailboxesQuasar if args.arch == "quasar" else Mailboxes

    print("=== LLK TRIAGE ===")
    print(f"Arch:     {args.arch}")
    print(f"Location: {args.location}")
    print(f"Device:   {args.device_id}")
    print()

    # Collect per-RISC debug state once up front. Two PC samples + the
    # status flags are shared between the mailbox classifier (verdict per
    # KERNEL_STARTED thread) and the dedicated `RISC debug state` section.
    try:
        risc_states = _collect_risc_states(
            args.arch, args.location, args.device_id, check_context, convert_coordinate
        )
    except Exception as exc:
        print(f"[llk-triage] failed to collect per-RISC state: {exc}", file=sys.stderr)
        risc_states = {}

    _print_mailbox_state(
        mailbox_cls, args.location, args.device_id, read_word_from_device, risc_states
    )
    print()
    _print_risc_states(risc_states)
    print()
    _print_tensix_cfg(args.location, args.device_id, get_tensix_state)
    print()
    print("=== END LLK TRIAGE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
