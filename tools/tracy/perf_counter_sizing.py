# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Size serialized counter packets, not just the number of counter records.

A TS_DATA_16B counter record contains an 8-byte timestamp plus 16 bytes of data:
three ordinary 8-byte marker slots. A DRAM program-support unit reserves only
six slots (two program-ID and four guaranteed markers). Every invocation between
profiler drains consumes capacity, including replays of the same program.

The C++ constants and hardware counter arrays are checked by CPU contract tests.
The optional-marker reserve is a workload assumption, not a proof of capture
completeness; inspect dropped-marker flags and increase it for heavily zoned ops.
"""

import math
from numbers import Integral, Real

from tracy.perf_counter_multipass import resolve_perf_counter_groups

# Number of counter records, mirroring hw_counters.h array lengths.
COUNTERS_PER_GROUP = {
    "blackhole": {
        "fpu": 3,
        "pack": 5,
        "unpack": 18,
        "l1_0": 16,
        "l1_1": 16,
        "instrn": 59,
        "l1_2": 16,
        "l1_3": 16,
        "l1_4": 16,
        "l1_5": 4,
    },
    "wormhole_b0": {
        "fpu": 3,
        "pack": 14,
        "unpack": 18,
        "l1_0": 16,
        "l1_1": 16,
        "instrn": 59,
    },
}

MARKER_BYTES = 8
COUNTER_PACKET_MARKER_SLOTS = 3
PROGRAM_SUPPORT_MARKER_SLOTS = 6
L1_OPTIONAL_MARKER_BUDGET = 250
L1_BUFFER_BYTES = (L1_OPTIONAL_MARKER_BUDGET + PROGRAM_SUPPORT_MARKER_SLOTS) * MARKER_BYTES
PROGRAM_SUPPORT_BYTES = PROGRAM_SUPPORT_MARKER_SLOTS * MARKER_BYTES
# C++ requires the DRAM allocation to be strictly larger than the L1 buffer.
MIN_PROGRAM_SUPPORT_COUNT = L1_BUFFER_BYTES // PROGRAM_SUPPORT_BYTES + 1
_DEFAULT_BRISC_RESERVE = 16
DISPATCH_HEADROOM_MARKER_SLOTS = 8
QUICK_PUSH_MARKER_SLOTS = 2
NOC_ALIGNMENT_MARKER_SLOTS = 2
QUICK_PUSH_GUARANTEED_END_SLOTS = 2


def normalized_counter_groups(arch, groups):
    """Use the capture CLI's aliases, all expansion and architecture validation."""
    if arch not in COUNTERS_PER_GROUP:
        raise ValueError(f"unknown sizing arch {arch!r}; known: {sorted(COUNTERS_PER_GROUP)}")
    return resolve_perf_counter_groups(groups, arch)


def counters_per_zone(arch, groups):
    """Counter records emitted per BRISC zone for the requested group set."""
    return sum(COUNTERS_PER_GROUP[arch][g] for g in normalized_counter_groups(arch, groups))


def marker_slots_per_zone(arch, groups):
    """Ordinary 8-byte slots occupied by the serialized counter records."""
    return COUNTER_PACKET_MARKER_SLOTS * counters_per_zone(arch, groups)


def markers_per_zone(arch, groups):
    """Alias for slot count; use counters_per_zone when counting CSV records."""
    return marker_slots_per_zone(arch, groups)


def _validate_reserve(brisc_reserve):
    if isinstance(brisc_reserve, bool) or not isinstance(brisc_reserve, Integral) or brisc_reserve < 0:
        raise ValueError("brisc_reserve must be a nonnegative integer marker-slot count")


def single_pass_l1_headroom(arch, groups, brisc_reserve=_DEFAULT_BRISC_RESERVE):
    """L1 slots left for one scheduled pass, including optional-marker reserve.

    Negative headroom requires BRISC's intermediate DRAM flushes; it is not by
    itself evidence of dropped markers. Check the complete DRAM budget as well.
    """
    _validate_reserve(brisc_reserve)
    return L1_OPTIONAL_MARKER_BUDGET - brisc_reserve - marker_slots_per_zone(arch, groups)


def _serialized_program_slots(counter_slots, brisc_reserve):
    # Reserve covers both regular optional markers and maximum live zone-stack
    # depth. Counter readout uses the stricter DISPATCH room check. A packet can
    # leave up to two unusable slots at the end of a chunk.
    payload_capacity = (
        L1_OPTIONAL_MARKER_BUDGET - DISPATCH_HEADROOM_MARKER_SLOTS - brisc_reserve - (COUNTER_PACKET_MARKER_SLOTS - 1)
    )
    if payload_capacity <= 0:
        raise ValueError("brisc_reserve leaves no safe counter-packet capacity")
    payload = counter_slots + brisc_reserve
    chunks = max(1, math.ceil(payload / payload_capacity))
    slots = payload + chunks * PROGRAM_SUPPORT_MARKER_SLOTS + (chunks - 1) * QUICK_PUSH_MARKER_SLOTS
    # Each chunk is aligned to 16 bytes. At most one slot is wasted per chunk;
    # the final total must also be aligned. Keep the guaranteed-end headroom
    # required by quick_push even if the final chunk is otherwise empty.
    slots += (chunks - 1) * (NOC_ALIGNMENT_MARKER_SLOTS - 1)
    slots = math.ceil(slots / NOC_ALIGNMENT_MARKER_SLOTS) * NOC_ALIGNMENT_MARKER_SLOTS
    if chunks > 1:
        slots += QUICK_PUSH_GUARANTEED_END_SLOTS
    return slots


def bytes_per_program(arch, groups, brisc_reserve=_DEFAULT_BRISC_RESERVE):
    """Estimated per-RISC bytes, including repeated flush headers and alignment.

    Assumes at most one counter zone per invocation and optional markers/live
    stack bounded by brisc_reserve. Flushes are assumed to occur when full;
    arbitrary explicit flushes or additional zones need a larger workload budget.
    """
    _validate_reserve(brisc_reserve)
    return _serialized_program_slots(marker_slots_per_zone(arch, groups), brisc_reserve) * MARKER_BYTES


def recommend_program_support_count(
    programs_per_device, safety=1.2, *, arch=None, groups=(), brisc_reserve=_DEFAULT_BRISC_RESERVE
):
    """Support units for invocations between drains in ONE scheduled pass.

    Specify arch/groups for counter captures. Omitting both sizes counter-free
    programs only. Pin the maximum over all passes across A/B runs, because this
    allocation enters the instrumented kernel build identity.
    """
    if (
        isinstance(programs_per_device, bool)
        or not isinstance(programs_per_device, Integral)
        or programs_per_device <= 0
    ):
        raise ValueError("programs_per_device must be a positive integer invocation count")
    if isinstance(safety, bool) or not isinstance(safety, Real) or not math.isfinite(safety) or safety < 1:
        raise ValueError("safety must be finite and at least 1")
    _validate_reserve(brisc_reserve)
    if arch is None:
        if groups:
            raise ValueError("arch is required when sizing counter groups")
        per_program = _serialized_program_slots(0, brisc_reserve) * MARKER_BYTES
    else:
        per_program = bytes_per_program(arch, groups, brisc_reserve)
    return max(
        MIN_PROGRAM_SUPPORT_COUNT,
        math.ceil(programs_per_device * per_program * safety / PROGRAM_SUPPORT_BYTES),
    )
