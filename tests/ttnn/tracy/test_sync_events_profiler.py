# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Unified sync events profiler test.

Single parameterized test that verifies, for each instrumented API:
1. Every expected event is present, on the expected RISC.
2. Payloads are valid -- CB ids are the CB the kernel used, local semaphore keys are
   L1 addresses, and remote semaphore payloads decode as tagged NoC addresses.
3. The blocking wait actually blocked for at least the producer's delay.

The remote-payload checks mirror SYNC_SIGNAL_NOC_ADDR in
tt_metal/tools/profiler/synchronization_event_profiler.hpp; if that encoding moves, the
decoder below has to move with it.
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

# Mirrors DELAY_CYCLES in tests/tt_metal/tools/profiler/test_sync_events.cpp: the producer
# spins that many `nop`s before signalling. Each nop retires in at least one cycle, so a
# consumer that really blocked cannot have waited less than this. There is deliberately no
# upper bound -- NoC and scheduling latency make one flaky, and an over-long wait is not a
# defect in the instrumentation this test covers.
DELAY_CYCLES = 10000
MIN_BLOCKING_CYCLES = DELAY_CYCLES

# A wait shorter than this is treated as non-blocking (the "instant" reserve/wait that some
# sequences perform before the blocking one).
NONBLOCKING_CYCLES = 1000

L1_MAX = 0x200000

# Legacy timer_id -> event name mapping (kSyncNames in streaming_profiler_zone_csv.cpp)
SYNC_LEGACY_IDS = {
    1000: "SYNC-CB-PUSH",
    1003: "SYNC-SEM-SET",
    1004: "SYNC-SEM-SET-REMOTE",
    1007: "SYNC-SEM-WAIT-KEY",
    1008: "SYNC-CB-WAIT-KEY",
    1009: "SYNC-CB-RESERVE-KEY",
    1010: "SYNC-CB-POP",
}

# --- NoC address decoding ---------------------------------------------------------------
# Mirrors SYNC_SIGNAL_NOC_ADDR and the NOC_XY_ADDR / NOC_MULTICAST_ADDR macros in
# tt_metal/hw/inc/internal/tt-1xx/*/noc/noc_parameters.h.
NOC_TAG_PRESENT = 1 << 63  # bit 63: "this payload carries a NoC index"
NOC_TAG_INDEX = 1 << 62  # bit 62: the NoC index itself
NOC_ADDR_LOCAL_BITS = 36
NOC_ADDR_NODE_ID_BITS = 6
LOCAL_MASK = (1 << NOC_ADDR_LOCAL_BITS) - 1
NODE_MASK = (1 << NOC_ADDR_NODE_ID_BITS) - 1


def decode_noc_tag(payload: int) -> tuple[Optional[int], int]:
    """Split a tagged payload into (noc_index, bare_address).

    Returns (None, payload) when the tag is absent, which after the all-or-nothing rule in
    synchronization_event_profiler.hpp means the capture came from an untagged build.
    """
    if not payload & NOC_TAG_PRESENT:
        return None, payload
    noc = 1 if payload & NOC_TAG_INDEX else 0
    return noc, payload & ~(NOC_TAG_PRESENT | NOC_TAG_INDEX)


def decode_noc_addr(addr: int) -> dict:
    """Decode a bare (untagged) NoC address into its fields.

    A unicast address sets only x,y (bits 36-47); a multicast descriptor also carries the
    start corner (bits 48-59), so anything above bit 47 means multicast.
    """
    local = addr & LOCAL_MASK
    coords = addr >> NOC_ADDR_LOCAL_BITS
    if coords < (1 << (2 * NOC_ADDR_NODE_ID_BITS)):
        return {
            "kind": "unicast",
            "l1": local,
            "x": coords & NODE_MASK,
            "y": (coords >> NOC_ADDR_NODE_ID_BITS) & NODE_MASK,
        }
    return {
        "kind": "multicast",
        "l1": local,
        "end_x": coords & NODE_MASK,
        "end_y": (coords >> NOC_ADDR_NODE_ID_BITS) & NODE_MASK,
        "start_x": (coords >> (2 * NOC_ADDR_NODE_ID_BITS)) & NODE_MASK,
        "start_y": (coords >> (3 * NOC_ADDR_NODE_ID_BITS)) & NODE_MASK,
    }


@dataclass
class SyncEvent:
    zone_name: str
    core_x: int
    core_y: int
    risc: str
    duration_cycles: int
    payload: Optional[int] = None
    timestamp: int = 0  # For ordering events


def parse_zone_csv(csv_path: Path) -> list[SyncEvent]:
    """Parse zone CSV and extract sync events.

    Handles two types of events:
    1. Zones (ZONE_START/ZONE_END): Have explicit zone names like SYNC-CB-RESERVE
    2. Signals (TS_DATA): Have empty zone names but timer_id maps to event name

    A zone carries timing and no payload; the matching `-KEY` signal carries the payload.
    Both are returned -- callers that only want timing filter on the `-KEY` suffix.
    """
    events = []
    zone_starts = {}  # Track zone starts to compute duration

    with open(csv_path) as f:
        # Skip first line (metadata: ARCH, CHIP_FREQ, etc.)
        first_line = f.readline()
        if not first_line.startswith("ARCH:") and not first_line.startswith("PCIe"):
            # Reset if it wasn't metadata
            f.seek(0)

        reader = csv.DictReader(f)
        for row in reader:
            # Handle column names with leading spaces (CSV quirk)
            def get_col(name):
                return row.get(name, row.get(f" {name}", "")).strip()

            zone_name = get_col("zone name")
            row_type = get_col("type")
            try:
                timer_id = int(get_col("timer_id") or 0)
                timestamp = int(get_col("time[cycles since reset]") or 0)
            except ValueError:
                continue

            # Only TS_DATA rows carry a payload. ZoneCsvConsumer leaves Row::data at its 0
            # default for zone rows and prints it unconditionally, so a "0" in a zone row
            # means "no payload" -- while in a TS_DATA row 0 is a real value (CB id 0 is the
            # CB these kernels use). Reading the column for both would collapse the two.
            data_str = get_col("data")
            payload = None
            if data_str and row_type == "TS_DATA":
                try:
                    payload = int(data_str, 0)
                except ValueError:
                    payload = None

            try:
                # Handle zones (ZONE_START/ZONE_END pairs); their payload rides in the
                # matching -KEY signal, not in the zone row.
                if zone_name.startswith("SYNC-"):
                    if row_type == "ZONE_START":
                        zone_starts[(zone_name, timer_id)] = timestamp
                    elif row_type == "ZONE_END":
                        start_ts = zone_starts.get((zone_name, timer_id), timestamp)
                        events.append(
                            SyncEvent(
                                zone_name=zone_name,
                                core_x=int(get_col("core_x") or 0),
                                core_y=int(get_col("core_y") or 0),
                                risc=get_col("RISC processor type"),
                                duration_cycles=timestamp - start_ts,
                                payload=None,
                                timestamp=start_ts,
                            )
                        )

                # Handle signals (TS_DATA with timer_id mapping)
                elif row_type == "TS_DATA" and timer_id in SYNC_LEGACY_IDS:
                    events.append(
                        SyncEvent(
                            zone_name=SYNC_LEGACY_IDS[timer_id],
                            core_x=int(get_col("core_x") or 0),
                            core_y=int(get_col("core_y") or 0),
                            risc=get_col("RISC processor type"),
                            duration_cycles=0,  # Signals are instantaneous
                            payload=payload,
                            timestamp=timestamp,
                        )
                    )

            except (ValueError, KeyError):
                continue

    # Sort by timestamp for proper ordering
    events.sort(key=lambda e: e.timestamp)
    return events


@dataclass
class ApiTest:
    """One row of the C++ test table in tests/tt_metal/tools/profiler/test_sync_events.cpp.

    `test_id` is the index into that table and is passed to the binary as argv[1], so the two
    tables must stay in the same order. Append new cases at the end of both.
    """

    test_id: int
    name: str
    # (event_name, risc_substring) pairs that must all appear
    expected_events: list
    wait_event: str  # the zone whose blocking duration is checked
    payload_type: str  # "cb_id" or "sem_addr"
    expected_cb_id: Optional[int] = None
    requires_quasar: bool = False
    # To check the noc that the remote semaphore is sent on.
    expected_noc: Optional[int] = None
    # To check remote semaphore is unicast or multicast
    remote_kind: Optional[str] = None
    # To check the multicast rectangle spans correct number of cores
    mcast_cores: int = 1


# Producers on BRISC default to NoC 0; producers on NCRISC default to NoC 1. The Semaphore
# class cases construct `Noc noc(0)` explicitly, so they stay on NoC 0 wherever they run.
API_TESTS = [
    # ========== Raw CB APIs ==========
    # CB wait: producer (BRISC) reserve+push, consumer (NCRISC) wait
    ApiTest(
        0,
        "cb_wait",
        [
            ("SYNC-CB-RESERVE", "BRISC"),  # instant reserve
            ("SYNC-CB-PUSH", "BRISC"),  # after delay, releases consumer
            ("SYNC-CB-WAIT", "NCRISC"),  # blocking wait completes
        ],
        "SYNC-CB-WAIT",
        "cb_id",
        expected_cb_id=0,
    ),
    # CB reserve: producer reserve+push+reserve(blocks), consumer wait+pop
    ApiTest(
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
        expected_cb_id=0,
    ),
    # ========== Raw Semaphore APIs ==========
    # noc_semaphore_set + noc_semaphore_wait
    ApiTest(
        2,
        "raw_sem_set_wait",
        [("SYNC-SEM-SET", "BRISC"), ("SYNC-SEM-WAIT", "NCRISC")],
        "SYNC-SEM-WAIT",
        "sem_addr",
    ),
    # noc_semaphore_inc (remote)
    ApiTest(
        3,
        "raw_sem_inc_remote",
        [("SYNC-SEM-SET-REMOTE", "BRISC"), ("SYNC-SEM-WAIT", "NCRISC")],
        "SYNC-SEM-WAIT",
        "sem_addr",
        expected_noc=0,
        remote_kind="unicast",
    ),
    # noc_semaphore_set + noc_semaphore_wait_min
    ApiTest(
        4,
        "raw_sem_wait_min",
        [("SYNC-SEM-SET", "BRISC"), ("SYNC-SEM-WAIT", "NCRISC")],
        "SYNC-SEM-WAIT",
        "sem_addr",
    ),
    # noc_semaphore_inc_multicast
    ApiTest(
        5,
        "raw_sem_inc_multicast",
        [("SYNC-SEM-SET-REMOTE", "BRISC"), ("SYNC-SEM-WAIT", "NCRISC")],
        "SYNC-SEM-WAIT",
        "sem_addr",
        expected_noc=0,
        remote_kind="multicast",
    ),
    # noc_semaphore_set_multicast
    ApiTest(
        6,
        "raw_sem_set_multicast",
        [("SYNC-SEM-SET-REMOTE", "BRISC"), ("SYNC-SEM-WAIT", "NCRISC")],
        "SYNC-SEM-WAIT",
        "sem_addr",
        expected_noc=0,
        remote_kind="multicast",
    ),
    # The remaining SYNC_SIGNAL_NOC_ADDR emitters, then the same raw remote APIs on NoC 1
    # (driven from NCRISC); without those the noc-index bit is only ever observed as 0.
    ApiTest(
        7,
        "raw_sem_set_remote",
        [("SYNC-SEM-SET-REMOTE", "BRISC"), ("SYNC-SEM-WAIT", "NCRISC")],
        "SYNC-SEM-WAIT",
        "sem_addr",
        expected_noc=0,
        remote_kind="unicast",
    ),
    ApiTest(
        8,
        "raw_sem_set_multicast_loopback_src",
        [("SYNC-SEM-SET-REMOTE", "BRISC"), ("SYNC-SEM-WAIT", "NCRISC")],
        "SYNC-SEM-WAIT",
        "sem_addr",
        expected_noc=0,
        remote_kind="multicast",
        mcast_cores=2,  # loopback includes the sender, so the rectangle spans both cores
    ),
    ApiTest(
        9,
        "raw_sem_inc_remote_noc1",
        [("SYNC-SEM-SET-REMOTE", "NCRISC"), ("SYNC-SEM-WAIT", "BRISC")],
        "SYNC-SEM-WAIT",
        "sem_addr",
        expected_noc=1,
        remote_kind="unicast",
    ),
    ApiTest(
        10,
        "raw_sem_inc_multicast_noc1",
        [("SYNC-SEM-SET-REMOTE", "NCRISC"), ("SYNC-SEM-WAIT", "BRISC")],
        "SYNC-SEM-WAIT",
        "sem_addr",
        expected_noc=1,
        remote_kind="multicast",
    ),
    ApiTest(
        11,
        "raw_sem_set_multicast_noc1",
        [("SYNC-SEM-SET-REMOTE", "NCRISC"), ("SYNC-SEM-WAIT", "BRISC")],
        "SYNC-SEM-WAIT",
        "sem_addr",
        expected_noc=1,
        remote_kind="multicast",
    ),
    # ========== Semaphore Class APIs ==========
    # Semaphore::set() + Semaphore::wait()
    ApiTest(
        12,
        "class_set_wait",
        [("SYNC-SEM-SET", "BRISC"), ("SYNC-SEM-WAIT", "NCRISC")],
        "SYNC-SEM-WAIT",
        "sem_addr",
    ),
    # Semaphore::up() + Semaphore::wait_min()
    ApiTest(
        13,
        "class_up_wait_min",
        [("SYNC-SEM-SET", "BRISC"), ("SYNC-SEM-WAIT", "NCRISC")],
        "SYNC-SEM-WAIT",
        "sem_addr",
    ),
    # Semaphore::up() remote
    ApiTest(
        14,
        "class_up_remote",
        [("SYNC-SEM-SET-REMOTE", "BRISC"), ("SYNC-SEM-WAIT", "NCRISC")],
        "SYNC-SEM-WAIT",
        "sem_addr",
        expected_noc=0,
        remote_kind="unicast",
    ),
    # Semaphore::set() + Semaphore::down() (down = wait + decrement)
    ApiTest(
        15,
        "class_set_down",
        [
            ("SYNC-SEM-SET", "BRISC"),  # set()
            ("SYNC-SEM-WAIT", "NCRISC"),  # down() wait part
            ("SYNC-SEM-SET", "NCRISC"),  # down() decrement part
        ],
        "SYNC-SEM-WAIT",
        "sem_addr",
    ),
    # Semaphore::set_multicast()
    ApiTest(
        16,
        "class_set_multicast",
        [
            ("SYNC-SEM-SET", "BRISC"),  # local set first
            ("SYNC-SEM-SET-REMOTE", "BRISC"),
            ("SYNC-SEM-WAIT", "NCRISC"),
        ],
        "SYNC-SEM-WAIT",
        "sem_addr",
        expected_noc=0,
        remote_kind="multicast",
    ),
    # Semaphore::inc_multicast()
    ApiTest(
        17,
        "class_inc_multicast",
        [("SYNC-SEM-SET-REMOTE", "BRISC"), ("SYNC-SEM-WAIT", "NCRISC")],
        "SYNC-SEM-WAIT",
        "sem_addr",
        expected_noc=0,
        remote_kind="multicast",
    ),
    # ========== Compute RISC (TRISC) CB APIs - all architectures ==========
    ApiTest(
        18,
        "compute_cb_brisc_push_trisc_wait",
        [
            ("SYNC-CB-RESERVE", "BRISC"),  # instant reserve
            ("SYNC-CB-PUSH", "BRISC"),  # after delay
            ("SYNC-CB-WAIT", "TRISC"),  # blocking wait
            ("SYNC-CB-POP", "TRISC"),  # pop
        ],
        "SYNC-CB-WAIT",
        "cb_id",
        expected_cb_id=0,
    ),
    ApiTest(
        19,
        "compute_cb_trisc_push_ncrisc_wait",
        [
            ("SYNC-CB-RESERVE", "TRISC"),  # instant reserve
            ("SYNC-CB-PUSH", "TRISC"),  # after delay
            ("SYNC-CB-WAIT", "NCRISC"),  # blocking wait
        ],
        "SYNC-CB-WAIT",
        "cb_id",
        expected_cb_id=0,
    ),
    # ========== Compute RISC (TRISC) Semaphore APIs - Quasar only ==========
    ApiTest(
        20,
        "compute_brisc_set_trisc_wait",
        [("SYNC-SEM-SET", "BRISC"), ("SYNC-SEM-WAIT", "TRISC0")],
        "SYNC-SEM-WAIT",
        "sem_addr",
        requires_quasar=True,
    ),
    ApiTest(
        21,
        "compute_brisc_set_trisc_wait_min",
        [("SYNC-SEM-SET", "BRISC"), ("SYNC-SEM-WAIT", "TRISC0")],
        "SYNC-SEM-WAIT",
        "sem_addr",
        requires_quasar=True,
    ),
    ApiTest(
        22,
        "compute_brisc_set_trisc_down",
        [
            ("SYNC-SEM-SET", "BRISC"),
            ("SYNC-SEM-WAIT", "TRISC0"),  # down() wait part
            ("SYNC-SEM-SET", "TRISC0"),  # down() decrement part
        ],
        "SYNC-SEM-WAIT",
        "sem_addr",
        requires_quasar=True,
    ),
    # TRISC producer + NCRISC consumer
    ApiTest(
        23,
        "compute_trisc_set_ncrisc_wait",
        [("SYNC-SEM-SET", "TRISC0"), ("SYNC-SEM-WAIT", "NCRISC")],
        "SYNC-SEM-WAIT",
        "sem_addr",
        requires_quasar=True,
    ),
    ApiTest(
        24,
        "compute_trisc_up_ncrisc_wait",
        [("SYNC-SEM-SET", "TRISC0"), ("SYNC-SEM-WAIT", "NCRISC")],
        "SYNC-SEM-WAIT",
        "sem_addr",
        requires_quasar=True,
    ),
]


def get_test_binary() -> Path:
    return TT_METAL_HOME / "build/test/tt_metal/tools/profiler/test_sync_events"


# What the C++ harness prints for a case it will not run on this device. Detecting the skip
# from the harness' own output keeps the arch check in one place: opening a device from here
# just to read `arch()` costs a full device init per session and gets the answer second-hand.
QUASAR_SKIP_MARKER = "requires Quasar"


def run_test(test_id: int, csv_path: Path) -> tuple[bool, str, bool]:
    """Run C++ test for a specific API.

    Returns: (success, log, streaming_profiler_active)
    """
    test_binary = get_test_binary()
    if not test_binary.exists():
        return False, f"Binary not found: {test_binary}", False

    env = os.environ.copy()
    env.update(
        {
            "TT_METAL_HOME": str(TT_METAL_HOME),
            "TT_METAL_STREAMING_PROFILER": "1",
            "TT_METAL_STREAMING_PROFILER_SYNC_EVENTS": "1",
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
    log = proc.stdout + proc.stderr
    streaming_active = "[streaming profiler] active" in log
    return proc.returncode == 0, log, streaming_active


def check_remote_payload(event: SyncEvent, spec: ApiTest) -> list[str]:
    """Validate one SYNC-SEM-SET-REMOTE payload against the tagged-NoC-address encoding."""
    errors = []
    noc, addr = decode_noc_tag(event.payload)
    if noc is None:
        return [
            f"{event.zone_name}: payload {hex(event.payload)} carries no NoC tag; "
            "every NoC address passed to SYNC_SIGNAL must go through SYNC_SIGNAL_NOC_ADDR"
        ]
    if noc != spec.expected_noc:
        errors.append(f"{event.zone_name}: NoC index {noc}, expected {spec.expected_noc}")

    fields = decode_noc_addr(addr)
    if spec.remote_kind is not None and fields["kind"] != spec.remote_kind:
        errors.append(f"{event.zone_name}: decoded as {fields['kind']}, expected {spec.remote_kind}")

    if not 0 < fields["l1"] <= L1_MAX:
        errors.append(f"{event.zone_name}: L1 offset {hex(fields['l1'])} outside (0, {hex(L1_MAX)}]")

    if fields["kind"] == "multicast":
        # The rectangle must round-trip to the size the kernel asked for; a mis-encoded
        # descriptor shows up here as a wrong core count.
        width = fields["end_x"] - fields["start_x"] + 1
        height = fields["end_y"] - fields["start_y"] + 1
        if width * height != spec.mcast_cores:
            rect = f"start=({fields['start_x']},{fields['start_y']}) end=({fields['end_x']},{fields['end_y']})"
            errors.append(f"{event.zone_name}: expected {spec.mcast_cores} core(s) in the rectangle, got {rect}")
    print(f"  ✓ {event.zone_name}: noc={noc} {fields}")
    return errors


@pytest.fixture(autouse=True)
def setup():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if not get_test_binary().exists():
        pytest.skip("Build test first: cmake --build build --target profiler_test_sync_events")


@pytest.mark.parametrize("spec", API_TESTS, ids=lambda s: s.name)
def test_sync_api(spec: ApiTest):
    """Unified test for each sync API. Verifies:
    1. Every expected event is present, on the expected RISC.
    2. Payloads are valid -- CB ids match the CB the kernel used, local semaphore keys are L1
       addresses, and remote payloads decode as tagged NoC addresses on the expected NoC.
    3. The blocking wait blocked for at least the producer's delay.
    """
    csv_path = ARTIFACTS / f"{spec.name}.csv"
    csv_path.unlink(missing_ok=True)

    success, log, streaming_active = run_test(spec.test_id, csv_path)
    assert success, f"Test failed:\n{log[-2000:]}"
    if spec.requires_quasar and QUASAR_SKIP_MARKER in log:
        pytest.skip("Test requires Quasar device (ckernel::Semaphore is Quasar-only)")
    if not streaming_active:
        pytest.skip("Streaming profiler did not activate (requires Blackhole with ENABLE_TRACY build)")
    assert csv_path.exists(), f"Zone CSV not written. Log:\n{log[-1000:]}"

    events = parse_zone_csv(csv_path)
    # Zones carry timing; the matching -KEY signals carry the payload.
    sync_events = [e for e in events if not e.zone_name.endswith("-KEY")]
    key_events = [e for e in events if e.zone_name.endswith("-KEY")]

    print(f"\n{'='*50}")
    print(f"API: {spec.name}")
    print(f"{'='*50}")

    # 1. Every expected event is present (order varies with concurrent execution)
    print("\nEvent presence verification:")
    print(f"  Expected: {len(spec.expected_events)} events")
    print(f"  Actual:   {len(sync_events)} events")

    actual_events = [(e.zone_name, e.risc.split()[0].upper() if e.risc else "") for e in sync_events]

    sequence_errors = []
    for expected_event, expected_risc in spec.expected_events:
        found = any(
            actual_name == expected_event and expected_risc.upper() in actual_risc.upper()
            for actual_name, actual_risc in actual_events
        )
        if found:
            print(f"  ✓ {expected_event} on {expected_risc}")
        else:
            print(f"  ✗ MISSING: {expected_event} on {expected_risc}")
            sequence_errors.append(f"MISSING: {expected_event} on {expected_risc}")

    if len(sync_events) > len(spec.expected_events):
        print(f"  (Also found {len(sync_events) - len(spec.expected_events)} additional events)")

    assert not sequence_errors, "Missing events:\n" + "\n".join(sequence_errors)

    # 2. Payloads. Remote events carry a tagged NoC address; everything else carries a local
    # key (a CB id or an L1 semaphore address), including the -KEY signals.
    print(f"\nPayload validation ({spec.payload_type}):")
    payload_errors = []
    sem_addresses = set()
    remote_seen = 0

    for e in sync_events + key_events:
        if e.payload is None:
            continue  # zone rows carry no payload; their -KEY signal does

        if e.zone_name == "SYNC-SEM-SET-REMOTE":
            remote_seen += 1
            payload_errors.extend(check_remote_payload(e, spec))
            continue

        if spec.payload_type == "cb_id":
            if not 0 <= e.payload < 64:
                payload_errors.append(f"{e.zone_name}: invalid CB ID {e.payload}")
            elif spec.expected_cb_id is not None and e.payload != spec.expected_cb_id:
                payload_errors.append(f"{e.zone_name}: expected CB ID {spec.expected_cb_id}, got {e.payload}")
        elif spec.payload_type == "sem_addr":
            if 0 < e.payload <= L1_MAX:
                sem_addresses.add(e.payload)
            else:
                payload_errors.append(f"{e.zone_name}: invalid local semaphore address {hex(e.payload)}")

    if spec.expected_noc is not None:
        assert remote_seen > 0, "expected at least one SYNC-SEM-SET-REMOTE payload, found none"

    for msg in payload_errors:
        print(f"  ✗ {msg}")
    if not payload_errors:
        if spec.payload_type == "cb_id":
            print(f"  ✓ All payloads have CB ID = {spec.expected_cb_id}")
        else:
            print(f"  ✓ Local semaphore addresses: {[hex(a) for a in sorted(sem_addresses)]}")

    assert not payload_errors, "Invalid payloads:\n" + "\n".join(payload_errors)

    # 3. The blocking wait must actually have blocked.
    wait_events = [e for e in sync_events if e.zone_name == spec.wait_event]
    assert wait_events, f"No {spec.wait_event} events found"

    blocking = [e for e in wait_events if e.duration_cycles > NONBLOCKING_CYCLES]
    assert blocking, (
        f"{len(wait_events)} {spec.wait_event} events found but none blocked "
        f"(> {NONBLOCKING_CYCLES} cycles); the consumer never waited on the producer"
    )

    durations = [e.duration_cycles for e in blocking]
    print(f"\nTiming ({spec.wait_event}):")
    print(f"  At least: {MIN_BLOCKING_CYCLES} cycles")
    print(f"  Measured: min={min(durations)}, max={max(durations)}, avg={sum(durations)/len(durations):.0f}")

    too_short = [d for d in durations if d < MIN_BLOCKING_CYCLES]
    assert not too_short, (
        f"{spec.wait_event} blocked for {too_short} cycles, less than the producer's "
        f"{DELAY_CYCLES}-nop delay; the wait did not cover the delay"
    )
    print(f"  ✓ {len(durations)}/{len(durations)} blocked at least {MIN_BLOCKING_CYCLES} cycles")


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
            "TT_METAL_STREAMING_PROFILER_SYNC_EVENTS": "0",  # Disabled
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
    log = proc.stdout + proc.stderr

    if proc.returncode != 0:
        pytest.skip(f"Test binary failed: {log[-500:]}")
    if "[streaming profiler] active" not in log:
        pytest.skip("Streaming profiler did not activate")
    if not csv_path.exists():
        pytest.skip("Zone CSV not written (streaming profiler may have no zones)")

    events = parse_zone_csv(csv_path)
    assert len(events) == 0, f"Found {len(events)} SYNC-* events when disabled"
    print("\n✓ No SYNC-* events when disabled")


def test_full_coverage_summary():
    """Run every API test in one capture and assert all event types are exercised."""
    csv_path = ARTIFACTS / "full.csv"
    csv_path.unlink(missing_ok=True)

    test_binary = get_test_binary()
    env = os.environ.copy()
    env.update(
        {
            "TT_METAL_HOME": str(TT_METAL_HOME),
            "TT_METAL_STREAMING_PROFILER": "1",
            "TT_METAL_STREAMING_PROFILER_SYNC_EVENTS": "1",
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
    log = proc.stdout + proc.stderr

    assert proc.returncode == 0, f"Test failed:\n{log[-1000:]}"
    if "[streaming profiler] active" not in log:
        pytest.skip("Streaming profiler did not activate (requires Blackhole with ENABLE_TRACY build)")
    assert csv_path.exists(), f"Zone CSV not written. Log:\n{log[-1000:]}"

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
    for event_name in sorted(by_type):
        status = "✓" if event_name in all_expected or event_name.endswith("-KEY") else "?"
        print(f"  {status} {event_name}: {by_type[event_name]}")

    covered = found & all_expected
    print(f"\nCoverage: {len(covered)}/{len(all_expected)} ({len(covered) / len(all_expected) * 100:.0f}%)")

    missing = all_expected - found
    assert not missing, f"Event types never emitted: {sorted(missing)}"
    print("✓ All event types covered")

    remote = [e for e in events if e.zone_name == "SYNC-SEM-SET-REMOTE" and e.payload is not None]
    assert remote, "no SYNC-SEM-SET-REMOTE payloads in the full capture"

    # The all-or-nothing rule, checked over the whole capture rather than per case: the presence
    # flag only lets a consumer fall back on an untagged build if EVERY emitter tags. One
    # SYNC_SIGNAL where SYNC_SIGNAL_NOC_ADDR belonged breaks that for the entire format, so this
    # runs across every event the run produced, not just the ones a case declared a NoC for.
    untagged = [e for e in remote if decode_noc_tag(e.payload)[0] is None]
    assert not untagged, (
        f"{len(untagged)}/{len(remote)} SYNC-SEM-SET-REMOTE payloads carry no NoC tag "
        f"(e.g. {hex(untagged[0].payload)} on {untagged[0].risc}); every NoC address passed to "
        "SYNC_SIGNAL must go through SYNC_SIGNAL_NOC_ADDR"
    )
    print(f"✓ All {len(remote)} SYNC-SEM-SET-REMOTE payloads carry a NoC tag")

    # Both NoC indices must appear, or the noc-index bit is untested.
    nocs = {decode_noc_tag(e.payload)[0] for e in remote}
    assert nocs == {0, 1}, f"expected SYNC-SEM-SET-REMOTE on both NoCs, saw {sorted(nocs)}"
    print("✓ SYNC-SEM-SET-REMOTE observed on both NoC 0 and NoC 1")
