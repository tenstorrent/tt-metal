# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Derive the perf-counter marker footprint and DRAM buffer size from the counter
group selection — so the profiler buffer is sized to fit, not guessed at.

Each enabled counter group emits one marker per counter per zone on the BRISC
read loop (perf_counters.hpp ``read_single_group``). The per-group counts are
fixed per architecture in ``tt_metal/hw/inc/internal/tt-1xx/<arch>/hw_counters.h``;
they are mirrored here so host tooling can size the buffer without a device.

Two budgets matter:
  * L1 optional-marker budget (PROFILER_L1_OPTIONAL_MARKER_COUNT, 250): a single
    profiling pass that requests more counter markers than this for one zone
    overflows the per-program L1 vector and drops markers.
  * DRAM program-support count (TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT): must
    cover the number of distinct programs/zones per device across the whole run.

Keep these tables in sync with hw_counters.h if the HW counter set changes.
"""

# Per-group counter counts == markers/zone/group on BRISC. Mirrors
# NUM_*_COUNTERS in each arch's hw_counters.h.
MARKERS_PER_GROUP = {
    "blackhole": {
        "fpu": 3,
        "pack": 5,
        "unpack": 22,
        "l1_0": 16,
        "l1_1": 16,
        "instrn": 59,
        "l1_2": 16,
        "l1_3": 16,
        "l1_4": 16,
    },
    "wormhole_b0": {
        "fpu": 3,
        "pack": 14,
        "unpack": 22,
        "l1_0": 16,
        "l1_1": 16,
        "instrn": 59,
        "l1_2": 0,
        "l1_3": 0,
        "l1_4": 0,
    },
}

# "sfpu" rides the FPU register path; it selects the FPU group (see __main__.py).
_GROUP_ALIASES = {"sfpu": "fpu"}

# "all" in the CLI = a single pass capturing these groups (L1 bank 1 needs a
# second pass because L1 banks share one mux).
_ALL_GROUPS = ["fpu", "pack", "unpack", "l1_0", "instrn"]

# hostdevcommon/profiler_common.h PROFILER_L1_OPTIONAL_MARKER_COUNT.
L1_OPTIONAL_MARKER_BUDGET = 250

# Guaranteed (non-counter) BRISC markers a normal op emits per zone — zone
# start/end plus the op's own data-movement timestamps. A conservative reserve
# so headroom math does not falsely report a fit.
_DEFAULT_BRISC_RESERVE = 16


def _normalize_groups(groups):
    out = []
    for g in groups:
        gl = g.lower()
        if gl == "all":
            out.extend(_ALL_GROUPS)
        else:
            out.append(_GROUP_ALIASES.get(gl, gl))
    return out


def markers_per_zone(arch, groups):
    """Counter markers emitted per zone on BRISC for the given group selection."""
    if arch not in MARKERS_PER_GROUP:
        raise ValueError(f"unknown arch {arch!r}; known: {sorted(MARKERS_PER_GROUP)}")
    table = MARKERS_PER_GROUP[arch]
    total = 0
    for g in _normalize_groups(groups):
        if g not in table:
            raise ValueError(f"unknown counter group {g!r} for {arch}")
        total += table[g]
    return total


def single_pass_l1_headroom(arch, groups, brisc_reserve=_DEFAULT_BRISC_RESERVE):
    """L1 optional-marker slots left after this single-pass group selection.

    Negative means the selection overflows one zone's L1 vector and will drop
    markers — split the groups across passes.
    """
    return L1_OPTIONAL_MARKER_BUDGET - brisc_reserve - markers_per_zone(arch, groups)


def recommend_program_support_count(programs_per_device, safety=1.2):
    """Smallest TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT covering the workload.

    Sizes the DRAM marker buffer to hold every distinct program/zone on a
    device, with headroom. Pin one value across runs: changing it re-hashes
    every kernel (full instrumented recompile).
    """
    if programs_per_device <= 0:
        raise ValueError("programs_per_device must be positive")
    import math

    return max(programs_per_device, math.ceil(programs_per_device * safety))
