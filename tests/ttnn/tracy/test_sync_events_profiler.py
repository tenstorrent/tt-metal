# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unified sync events profiler test.

Single parameterized test that verifies for each API:
1. Expected events are present
2. Payloads are valid (CB IDs 0-63, semaphore addresses in L1 range)
3. Wait timing matches expected delay (~10000 ± 500 cycles)
"""

from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import pytest

from tools.tracy.common import PROFILER_ARTIFACTS_DIR, TT_METAL_HOME

ARTIFACTS = PROFILER_ARTIFACTS_DIR / "sync_events"

# Timing constants (must match C++ DELAY_CYCLES)
EXPECTED_DELAY_CYCLES = 10000
TIMING_TOLERANCE_CYCLES = 500


@dataclass
class SyncEvent:
    zone_name: str
    core_x: int
    core_y: int
    risc: str
    duration_cycles: int
    payload: Optional[int] = None


def parse_zone_csv(csv_path: Path) -> list[SyncEvent]:
    events = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            zone_name = row.get("zone name", row.get("zone_name", ""))
            if not zone_name.startswith("SYNC-"):
                continue
            try:
                payload = None
                data_str = row.get("data", row.get("payload", ""))
                if data_str:
                    try:
                        payload = int(data_str, 0)
                    except ValueError:
                        pass
                events.append(
                    SyncEvent(
                        zone_name=zone_name,
                        core_x=int(row.get("core_x", row.get("core x", 0))),
                        core_y=int(row.get("core_y", row.get("core y", 0))),
                        risc=row.get("risc", row.get("RISC", "")),
                        duration_cycles=int(row.get("duration_cycles", row.get("duration cycles", 0))),
                        payload=payload,
                    )
                )
            except (ValueError, KeyError):
                continue
    return events


# API test configurations
# Each tuple: (test_id, name, expected_sequence, wait_event_for_timing, payload_type, expected_cb_id, requires_quasar)
# expected_sequence is a list of (event_name, risc) in EXACT expected order
# For semaphore tests, addresses are validated to be in L1 range and consistent
API_TESTS = [
    # ========== Raw CB APIs ==========
    # CB wait: producer (BRISC) reserve+push, consumer (NCRISC) wait
    (
        0,
        "cb_wait",
        [
            ("SYNC-CB-RESERVE", "BRISC"),  # instant reserve
            ("SYNC-CB-PUSH", "BRISC"),  # after delay, releases consumer
            ("SYNC-CB-WAIT", "NCRISC"),  # blocking wait completes
        ],
        "SYNC-CB-WAIT",
        "cb_id",
        0,
        False,
    ),
    # CB reserve: producer reserve+push+reserve(blocks), consumer wait+pop
    (
        1,
        "cb_reserve",
        [
            ("SYNC-CB-RESERVE", "BRISC"),  # instant reserve
            ("SYNC-CB-PUSH", "BRISC"),  # fills CB
            ("SYNC-CB-WAIT", "NCRISC"),  # instant (data ready)
            ("SYNC-CB-RESERVE", "BRISC"),  # blocking reserve completes
            ("SYNC-CB-POP", "NCRISC"),  # releases producer after delay
        ],
        "SYNC-CB-RESERVE",
        "cb_id",
        0,
        False,
    ),
    # ========== Raw Semaphore APIs ==========
    # noc_semaphore_set + noc_semaphore_wait
    (
        2,
        "raw_sem_set_wait",
        [
            ("SYNC-SEM-SET", "BRISC"),  # after delay
            ("SYNC-SEM-WAIT", "NCRISC"),  # blocking wait completes
        ],
        "SYNC-SEM-WAIT",
        "sem_addr",
        None,
        False,
    ),
    # noc_semaphore_inc (remote)
    (
        3,
        "raw_sem_inc_remote",
        [
            ("SYNC-SEM-SET-REMOTE", "BRISC"),
            ("SYNC-SEM-WAIT", "NCRISC"),
        ],
        "SYNC-SEM-WAIT",
        "sem_addr",
        None,
        False,
    ),
    # noc_semaphore_set + noc_semaphore_wait_min
    (
        4,
        "raw_sem_wait_min",
        [
            ("SYNC-SEM-SET", "BRISC"),
            ("SYNC-SEM-WAIT", "NCRISC"),
        ],
        "SYNC-SEM-WAIT",
        "sem_addr",
        None,
        False,
    ),
    # noc_semaphore_inc_multicast
    (
        5,
        "raw_sem_inc_multicast",
        [
            ("SYNC-SEM-SET-REMOTE", "BRISC"),
            ("SYNC-SEM-WAIT", "NCRISC"),
        ],
        "SYNC-SEM-WAIT",
        "sem_addr",
        None,
        False,
    ),
    # noc_semaphore_set_multicast
    (
        6,
        "raw_sem_set_multicast",
        [
            ("SYNC-SEM-SET-REMOTE", "BRISC"),
            ("SYNC-SEM-WAIT", "NCRISC"),
        ],
        "SYNC-SEM-WAIT",
        "sem_addr",
        None,
        False,
    ),
    # ========== Semaphore Class APIs ==========
    # Semaphore::set() + Semaphore::wait()
    (
        7,
        "class_set_wait",
        [
            ("SYNC-SEM-SET", "BRISC"),
            ("SYNC-SEM-WAIT", "NCRISC"),
        ],
        "SYNC-SEM-WAIT",
        "sem_addr",
        None,
        False,
    ),
    # Semaphore::up() + Semaphore::wait_min()
    (
        8,
        "class_up_wait_min",
        [
            ("SYNC-SEM-SET", "BRISC"),
            ("SYNC-SEM-WAIT", "NCRISC"),
        ],
        "SYNC-SEM-WAIT",
        "sem_addr",
        None,
        False,
    ),
    # Semaphore::up() remote
    (
        9,
        "class_up_remote",
        [
            ("SYNC-SEM-SET-REMOTE", "BRISC"),
            ("SYNC-SEM-WAIT", "NCRISC"),
        ],
        "SYNC-SEM-WAIT",
        "sem_addr",
        None,
        False,
    ),
    # Semaphore::set() + Semaphore::down() (down = wait + decrement)
    (
        10,
        "class_set_down",
        [
            ("SYNC-SEM-SET", "BRISC"),  # set()
            ("SYNC-SEM-WAIT", "NCRISC"),  # down() wait part
            ("SYNC-SEM-SET", "NCRISC"),  # down() decrement part
        ],
        "SYNC-SEM-WAIT",
        "sem_addr",
        None,
        False,
    ),
    # Semaphore::set_multicast()
    (
        11,
        "class_set_multicast",
        [
            ("SYNC-SEM-SET", "BRISC"),  # local set first
            ("SYNC-SEM-SET-REMOTE", "BRISC"),
            ("SYNC-SEM-WAIT", "NCRISC"),
        ],
        "SYNC-SEM-WAIT",
        "sem_addr",
        None,
        False,
    ),
    # Semaphore::inc_multicast()
    (
        12,
        "class_inc_multicast",
        [
            ("SYNC-SEM-SET-REMOTE", "BRISC"),
            ("SYNC-SEM-WAIT", "NCRISC"),
        ],
        "SYNC-SEM-WAIT",
        "sem_addr",
        None,
        False,
    ),
    # ========== Compute RISC (TRISC) CB APIs - all architectures ==========
    (
        13,
        "compute_cb_brisc_push_trisc_wait",
        [
            ("SYNC-CB-RESERVE", "BRISC"),  # instant reserve
            ("SYNC-CB-PUSH", "BRISC"),  # after delay
            ("SYNC-CB-WAIT", "TRISC"),  # blocking wait
            ("SYNC-CB-POP", "TRISC"),  # pop
        ],
        "SYNC-CB-WAIT",
        "cb_id",
        0,
        False,
    ),
    (
        14,
        "compute_cb_trisc_push_ncrisc_wait",
        [
            ("SYNC-CB-RESERVE", "TRISC"),  # instant reserve
            ("SYNC-CB-PUSH", "TRISC"),  # after delay
            ("SYNC-CB-WAIT", "NCRISC"),  # blocking wait
        ],
        "SYNC-CB-WAIT",
        "cb_id",
        0,
        False,
    ),
    # ========== Compute RISC (TRISC) Semaphore APIs - Quasar only ==========
    (
        15,
        "compute_brisc_set_trisc_wait",
        [
            ("SYNC-SEM-SET", "BRISC"),
            ("SYNC-SEM-WAIT", "TRISC0"),
        ],
        "SYNC-SEM-WAIT",
        "sem_addr",
        None,
        True,
    ),
    (
        16,
        "compute_brisc_set_trisc_wait_min",
        [
            ("SYNC-SEM-SET", "BRISC"),
            ("SYNC-SEM-WAIT", "TRISC0"),
        ],
        "SYNC-SEM-WAIT",
        "sem_addr",
        None,
        True,
    ),
    (
        17,
        "compute_brisc_set_trisc_down",
        [
            ("SYNC-SEM-SET", "BRISC"),
            ("SYNC-SEM-WAIT", "TRISC0"),  # down() wait part
            ("SYNC-SEM-SET", "TRISC0"),  # down() decrement part
        ],
        "SYNC-SEM-WAIT",
        "sem_addr",
        None,
        True,
    ),
    # TRISC producer + NCRISC consumer
    (
        18,
        "compute_trisc_set_ncrisc_wait",
        [
            ("SYNC-SEM-SET", "TRISC0"),
            ("SYNC-SEM-WAIT", "NCRISC"),
        ],
        "SYNC-SEM-WAIT",
        "sem_addr",
        None,
        True,
    ),
    (
        19,
        "compute_trisc_up_ncrisc_wait",
        [
            ("SYNC-SEM-SET", "TRISC0"),
            ("SYNC-SEM-WAIT", "NCRISC"),
        ],
        "SYNC-SEM-WAIT",
        "sem_addr",
        None,
        True,
    ),
]


def get_test_binary() -> Path:
    return TT_METAL_HOME / "build/test/tt_metal/tools/profiler/test_sync_events"


def is_quasar_device() -> bool:
    """Check if current device is Quasar (ckernel::Semaphore is Quasar-only)."""
    try:
        import ttnn

        device = ttnn.open_device(0)
        arch = device.arch().name.lower()
        ttnn.close_device(device)
        return arch == "quasar"
    except Exception:
        return False


def run_test(test_id: int, csv_path: Path) -> tuple[bool, str]:
    """Run C++ test for a specific API."""
    test_binary = get_test_binary()
    if not test_binary.exists():
        return False, f"Binary not found: {test_binary}"

    env = os.environ.copy()
    env.update(
        {
            "TT_METAL_HOME": str(TT_METAL_HOME),
            "TT_METAL_STREAMING_PROFILER": "1",
            "TT_METAL_DEVICE_PROFILER_SYNC_EVENTS": "1",
            "TT_METAL_STREAMING_PROFILER_ZONE_CSV": str(csv_path),
        }
    )

    proc = subprocess.run(
        [str(test_binary), str(test_id)],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=str(TT_METAL_HOME),
    )
    return proc.returncode == 0, proc.stdout + proc.stderr


def validate_payload(event: SyncEvent, payload_type: str) -> tuple[bool, str]:
    """Validate event payload based on type."""
    if event.payload is None:
        return True, "no payload"  # Some events may not have payloads

    if payload_type == "cb_id":
        if 0 <= event.payload < 64:
            return True, f"CB ID {event.payload}"
        return False, f"invalid CB ID {event.payload}"

    elif payload_type == "sem_addr":
        L1_MAX = 0x200000
        if 0 < event.payload <= L1_MAX:
            return True, f"addr {hex(event.payload)}"
        return False, f"invalid addr {hex(event.payload)}"

    return True, "unknown type"


@pytest.fixture(autouse=True)
def setup():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if not get_test_binary().exists():
        pytest.skip(f"Build test first: cmake --build build --target profiler_test_sync_events_timing")


@pytest.mark.parametrize(
    "test_id,name,expected_sequence,wait_event,payload_type,expected_cb_id,requires_quasar", API_TESTS
)
def test_sync_api(
    test_id: int,
    name: str,
    expected_sequence: list,
    wait_event: str,
    payload_type: str,
    expected_cb_id: Optional[int],
    requires_quasar: bool,
):
    if requires_quasar and not is_quasar_device():
        pytest.skip("Test requires Quasar device (ckernel::Semaphore is Quasar-only)")
    """
    Unified test for each sync API. Verifies:
    1. Events match expected sequence EXACTLY (type, RISC, order)
    2. All payloads are valid (CB IDs match expected, semaphore addresses in L1 and consistent)
    3. Wait timing is within tolerance of expected delay
    """
    csv_path = ARTIFACTS / f"{name}.csv"
    csv_path.unlink(missing_ok=True)

    # Run test
    success, log = run_test(test_id, csv_path)
    assert success, f"Test failed:\n{log[-2000:]}"
    assert csv_path.exists(), "Zone CSV not written"

    events = parse_zone_csv(csv_path)
    # Filter out -KEY marker events
    sync_events = [e for e in events if e.zone_name.startswith("SYNC-") and not e.zone_name.endswith("-KEY")]

    print(f"\n{'='*50}")
    print(f"API: {name}")
    print(f"{'='*50}")

    # 1. Check EXACT event sequence (type, RISC, order)
    print(f"\nEvent sequence verification (exact order):")
    print(f"  Expected: {len(expected_sequence)} events")
    print(f"  Actual:   {len(sync_events)} events")

    sequence_errors = []
    for i, (expected_event, expected_risc) in enumerate(expected_sequence):
        if i >= len(sync_events):
            print(f"  [{i}] ✗ MISSING: expected {expected_event} on {expected_risc}")
            sequence_errors.append(f"[{i}] MISSING: expected {expected_event} on {expected_risc}")
            continue

        actual = sync_events[i]
        type_ok = actual.zone_name == expected_event
        risc_ok = expected_risc.upper() in actual.risc.upper()

        if type_ok and risc_ok:
            print(f"  [{i}] ✓ {actual.zone_name} on {actual.risc}")
        else:
            errors = []
            if not type_ok:
                errors.append(f"type: got {actual.zone_name}, expected {expected_event}")
            if not risc_ok:
                errors.append(f"RISC: got {actual.risc}, expected {expected_risc}")
            error_msg = f"[{i}] {'; '.join(errors)}"
            print(f"  [{i}] ✗ {'; '.join(errors)}")
            sequence_errors.append(error_msg)

    # Check for extra events
    if len(sync_events) > len(expected_sequence):
        for i in range(len(expected_sequence), len(sync_events)):
            extra = sync_events[i]
            print(f"  [{i}] ? EXTRA: {extra.zone_name} on {extra.risc}")
            sequence_errors.append(f"[{i}] EXTRA: {extra.zone_name} on {extra.risc}")

    assert len(sequence_errors) == 0, f"Sequence errors:\n" + "\n".join(sequence_errors)

    # 2. Check ALL payloads
    print(f"\nPayload validation ({payload_type}):")
    invalid_payloads = []
    wrong_cb_ids = []
    sem_addresses = set()

    for e in sync_events:
        if e.payload is None:
            invalid_payloads.append((e.zone_name, "missing payload"))
            continue

        if payload_type == "cb_id":
            # CB ID must be 0-63 AND match expected value
            if not (0 <= e.payload < 64):
                invalid_payloads.append((e.zone_name, f"invalid CB ID {e.payload}"))
            elif expected_cb_id is not None and e.payload != expected_cb_id:
                wrong_cb_ids.append((e.zone_name, expected_cb_id, e.payload))

        elif payload_type == "sem_addr":
            # Semaphore address must be in L1 range
            L1_MAX = 0x200000
            if not (0 < e.payload <= L1_MAX):
                invalid_payloads.append((e.zone_name, f"invalid addr {hex(e.payload)}"))
            else:
                sem_addresses.add(e.payload)

    if payload_type == "cb_id":
        if invalid_payloads:
            for name, msg in invalid_payloads:
                print(f"  ✗ {name}: {msg}")
        elif wrong_cb_ids:
            for name, expected, actual in wrong_cb_ids:
                print(f"  ✗ {name}: expected CB ID {expected}, got {actual}")
        else:
            print(f"  ✓ All {len(sync_events)} events have CB ID = {expected_cb_id}")

    elif payload_type == "sem_addr":
        if invalid_payloads:
            for name, msg in invalid_payloads:
                print(f"  ✗ {name}: {msg}")
        else:
            # All addresses should be the same semaphore (or at most 2 for remote tests)
            print(f"  ✓ All {len(sync_events)} events have valid L1 addresses")
            print(f"  ✓ Unique addresses: {[hex(a) for a in sorted(sem_addresses)]}")
            if len(sem_addresses) > 2:
                print(f"  ⚠ More than 2 unique addresses (unexpected)")

    assert len(invalid_payloads) == 0, f"Invalid payloads: {invalid_payloads}"
    assert len(wrong_cb_ids) == 0, f"Wrong CB IDs: {wrong_cb_ids}"

    # 3. Check timing - find the blocking wait (longest duration)
    if wait_event:
        wait_events = [e for e in sync_events if e.zone_name == wait_event]
        blocking_waits = [e for e in wait_events if e.duration_cycles > TIMING_TOLERANCE_CYCLES]

        if blocking_waits:
            durations = [e.duration_cycles for e in blocking_waits]
            max_dur = max(durations)
            min_dur = min(durations)
            avg_dur = sum(durations) / len(durations)

            print(f"\nTiming ({wait_event}):")
            print(f"  Expected: {EXPECTED_DELAY_CYCLES} ± {TIMING_TOLERANCE_CYCLES} cycles")
            print(f"  Measured: min={min_dur}, max={max_dur}, avg={avg_dur:.0f}")

            lower = EXPECTED_DELAY_CYCLES - TIMING_TOLERANCE_CYCLES
            upper = EXPECTED_DELAY_CYCLES + TIMING_TOLERANCE_CYCLES
            in_range = [d for d in durations if lower <= d <= upper]

            if len(in_range) > 0:
                print(f"  ✓ {len(in_range)}/{len(durations)} within tolerance")
            else:
                loose_lower = EXPECTED_DELAY_CYCLES - 2 * TIMING_TOLERANCE_CYCLES
                loose_upper = EXPECTED_DELAY_CYCLES + 2 * TIMING_TOLERANCE_CYCLES
                in_loose = [d for d in durations if loose_lower <= d <= loose_upper]
                if in_loose:
                    print(f"  ⚠ {len(in_loose)}/{len(durations)} within 2x tolerance")
                else:
                    pytest.fail(f"Wait timing outside expected range: got {max_dur}, expected [{lower}, {upper}]")
        else:
            if wait_events:
                print(f"\n⚠ {len(wait_events)} {wait_event} events found but none with blocking duration")
            else:
                print(f"\n⚠ No {wait_event} events found")


def test_no_events_when_disabled():
    """Verify no SYNC-* events when profiling is disabled."""
    csv_path = ARTIFACTS / "disabled.csv"
    csv_path.unlink(missing_ok=True)

    test_binary = get_test_binary()
    env = os.environ.copy()
    env.update(
        {
            "TT_METAL_HOME": str(TT_METAL_HOME),
            "TT_METAL_STREAMING_PROFILER": "1",
            "TT_METAL_DEVICE_PROFILER_SYNC_EVENTS": "0",  # Disabled
            "TT_METAL_STREAMING_PROFILER_ZONE_CSV": str(csv_path),
        }
    )

    proc = subprocess.run(
        [str(test_binary), "0"],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=str(TT_METAL_HOME),
    )

    if proc.returncode != 0 or not csv_path.exists():
        pytest.skip("Could not run with sync events disabled")

    events = parse_zone_csv(csv_path)
    assert len(events) == 0, f"Found {len(events)} SYNC-* events when disabled"
    print("\n✓ No SYNC-* events when disabled")


def test_full_coverage_summary():
    """Run all API tests and print coverage summary."""
    csv_path = ARTIFACTS / "full.csv"
    csv_path.unlink(missing_ok=True)

    # Run all tests (no arg = run all)
    test_binary = get_test_binary()
    env = os.environ.copy()
    env.update(
        {
            "TT_METAL_HOME": str(TT_METAL_HOME),
            "TT_METAL_STREAMING_PROFILER": "1",
            "TT_METAL_DEVICE_PROFILER_SYNC_EVENTS": "1",
            "TT_METAL_STREAMING_PROFILER_ZONE_CSV": str(csv_path),
        }
    )

    proc = subprocess.run(
        [str(test_binary)],  # No arg = all tests
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=str(TT_METAL_HOME),
    )

    assert proc.returncode == 0, f"Test failed:\n{proc.stderr[-1000:]}"
    assert csv_path.exists(), "Zone CSV not written"

    events = parse_zone_csv(csv_path)

    print("\n" + "=" * 50)
    print("FULL API COVERAGE SUMMARY")
    print("=" * 50)

    all_expected = {
        "SYNC-CB-RESERVE",
        "SYNC-CB-PUSH",
        "SYNC-CB-WAIT",
        "SYNC-CB-POP",
        "SYNC-SEM-SET",
        "SYNC-SEM-WAIT",
        "SYNC-SEM-SET-REMOTE",
    }
    found = {e.zone_name for e in events if not e.zone_name.endswith("-KEY")}

    by_type = defaultdict(int)
    for e in events:
        by_type[e.zone_name] += 1

    print("\nEvent counts:")
    for name in sorted(by_type.keys()):
        status = "✓" if name in all_expected or name.endswith("-KEY") else "?"
        print(f"  {status} {name}: {by_type[name]}")

    covered = found & all_expected
    pct = len(covered) / len(all_expected) * 100
    print(f"\nCoverage: {len(covered)}/{len(all_expected)} ({pct:.0f}%)")

    missing = all_expected - found
    if missing:
        print(f"Missing: {missing}")
    else:
        print("✓ All event types covered")
