# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Source-contract tests for counter wire size and profiler DRAM allocation."""

import math
import re
from pathlib import Path

import pytest

from tracy.perf_counter_sizing import (
    COUNTERS_PER_GROUP,
    COUNTER_PACKET_MARKER_SLOTS,
    DISPATCH_HEADROOM_MARKER_SLOTS,
    NOC_ALIGNMENT_MARKER_SLOTS,
    QUICK_PUSH_MARKER_SLOTS,
    L1_BUFFER_BYTES,
    L1_OPTIONAL_MARKER_BUDGET,
    MARKER_BYTES,
    MIN_PROGRAM_SUPPORT_COUNT,
    PROGRAM_SUPPORT_BYTES,
    PROGRAM_SUPPORT_MARKER_SLOTS,
    bytes_per_program,
    counters_per_zone,
    marker_slots_per_zone,
    recommend_program_support_count,
    single_pass_l1_headroom,
)

ROOT = Path(__file__).resolve().parents[3]


def _header_constants():
    text = (ROOT / "tt_metal/hw/inc/hostdev/profiler_common.h").read_text()
    return {
        name: int(value)
        for name, value in re.findall(r"constexpr static std::uint32_t (PROFILER_L1_\w+) = (\d+);", text)
    }


@pytest.mark.parametrize("arch,header_arch", [("blackhole", "blackhole"), ("wormhole_b0", "wormhole")])
def test_counter_table_matches_cpp_array_declarations(arch, header_arch):
    text = (ROOT / f"tt_metal/hw/inc/internal/tt-1xx/{header_arch}/hw_counters.h").read_text()
    # Count initialized records independently of NUM_* constants; some now use
    # array.size(), which made the old mirrored UNPACK=22 table silently stale.
    arrays = re.findall(
        r"std::array<std::pair<PerfCounterType,\s*std::uint16_t>,\s*\w+>\s+(\w+)_counters\s*=\s*\{(.*?)\};",
        text,
        re.S,
    )
    counts = {name: len(re.findall(r"PerfCounterType::", body)) for name, body in arrays}
    counts = {name: count for name, count in counts.items() if count}
    assert counts == COUNTERS_PER_GROUP[arch]


def test_packet_slots_follow_cpp_timestamped_data_layout():
    counter = (ROOT / "tt_metal/tools/profiler/perf_counters.hpp").read_text()
    kernel = (ROOT / "tt_metal/tools/profiler/kernel_profiler.hpp").read_text()
    assert re.search(r"PacketTypes::TS_DATA_16B>\(counter.raw_data_1, counter.raw_data_2\)", counter)
    assert "PROFILER_L1_MARKER_UINT32_SIZE * (2 + sizeof...(trailers))" in kernel
    words_per_marker = _header_constants()["PROFILER_L1_MARKER_UINT32_SIZE"]
    # One timestamp, first uint64 payload and one uint64 trailer.
    serialized_counter_bytes = words_per_marker * (2 + 1) * 4
    assert serialized_counter_bytes == 24
    assert COUNTER_PACKET_MARKER_SLOTS * MARKER_BYTES == serialized_counter_bytes


def test_support_unit_matches_cpp_allocation_and_strict_l1_minimum():
    constants = _header_constants()
    slots = constants["PROFILER_L1_PROGRAM_ID_COUNT"] + constants["PROFILER_L1_GUARANTEED_MARKER_COUNT"]
    marker_bytes = constants["PROFILER_L1_MARKER_UINT32_SIZE"] * 4
    assert (slots, marker_bytes) == (PROGRAM_SUPPORT_MARKER_SLOTS, MARKER_BYTES)
    assert PROGRAM_SUPPORT_BYTES == slots * marker_bytes == 48
    assert L1_OPTIONAL_MARKER_BUDGET == constants["PROFILER_L1_OPTIONAL_MARKER_COUNT"]
    assert L1_BUFFER_BYTES == (slots + L1_OPTIONAL_MARKER_BUDGET) * marker_bytes == 2048
    assert MIN_PROGRAM_SUPPORT_COUNT * PROGRAM_SUPPORT_BYTES > L1_BUFFER_BYTES
    assert (MIN_PROGRAM_SUPPORT_COUNT - 1) * PROGRAM_SUPPORT_BYTES <= L1_BUFFER_BYTES
    allocation = (ROOT / "tt_metal/impl/profiler/profiler_state_manager.cpp").read_text()
    assert "dram_bank_size_per_risc_bytes_single_program * profiler_program_support_count.value()" in allocation
    assert "TT_ASSERT(dram_bank_size_per_risc_bytes > kernel_profiler::PROFILER_L1_BUFFER_SIZE)" in allocation


def test_fpu_instrn_capture_fits_l1_but_needs_more_than_one_support_unit_per_op():
    groups = ["fpu", "instrn"]
    assert counters_per_zone("blackhole", groups) == 62
    assert marker_slots_per_zone("blackhole", groups) == 186
    assert single_pass_l1_headroom("blackhole", groups) == 48
    assert bytes_per_program("blackhole", groups) == 1664
    count = recommend_program_support_count(20, arch="blackhole", groups=groups)
    assert count == 832
    assert count * 48 >= math.ceil(20 * 1664 * 1.2)
    assert 512 * 48 < 20 * 1664  # The earlier proposed 512-unit capture can truncate.


def test_aliases_and_repeated_groups_do_not_double_count():
    assert counters_per_zone("blackhole", ["sfpu", "FPU", "fpu"]) == 3
    assert counters_per_zone("blackhole", ["all", "l1_1"]) == counters_per_zone("blackhole", ["all"])
    assert counters_per_zone("blackhole", ["l1_1", "all"]) == counters_per_zone("blackhole", ["all"])
    assert counters_per_zone("blackhole", ["l1_5"]) == 4
    assert counters_per_zone("wormhole_b0", ["pack"]) == 14


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_invalid_invocation_count_fails(count):
    with pytest.raises(ValueError, match="positive integer"):  # allow-pytest.raises: no device conftest
        recommend_program_support_count(count)


@pytest.mark.parametrize("safety", [0, 0.99, float("nan"), float("inf"), True])
def test_invalid_safety_fails(safety):
    with pytest.raises(ValueError, match="safety"):  # allow-pytest.raises: no device conftest
        recommend_program_support_count(20, safety)


@pytest.mark.parametrize("reserve", [-1, 1.5, True])
def test_invalid_optional_reserve_fails(reserve):
    with pytest.raises(ValueError, match="brisc_reserve"):  # allow-pytest.raises: no device conftest
        single_pass_l1_headroom("blackhole", ["fpu"], reserve)


@pytest.mark.parametrize("arch,groups", [("unknown", ["fpu"]), ("blackhole", ["typo"]), ("wormhole_b0", ["l1_5"])])
def test_unsupported_capture_cannot_silently_underestimate(arch, groups):
    with pytest.raises(ValueError):  # allow-pytest.raises: no device conftest
        counters_per_zone(arch, groups)


def test_counter_groups_require_architecture_and_small_allocations_cover_l1():
    with pytest.raises(ValueError, match="arch is required"):  # allow-pytest.raises: no device conftest
        recommend_program_support_count(20, groups=["fpu"])
    assert recommend_program_support_count(1, safety=1) == MIN_PROGRAM_SUPPORT_COUNT


def test_flush_overhead_follows_cpp_and_covers_large_pass_without_safety_margin():
    kernel = (ROOT / "tt_metal/tools/profiler/kernel_profiler.hpp").read_text()
    constants = {name: int(value) for name, value in re.findall(r"constexpr uint32_t (\w+) = (\d+);", kernel)}
    assert QUICK_PUSH_MARKER_SLOTS == constants["QUICK_PUSH_MARKER_COUNT"]
    assert NOC_ALIGNMENT_MARKER_SLOTS * 2 == constants["NOC_ALIGNMENT_FACTOR"]
    words = _header_constants()["PROFILER_L1_MARKER_UINT32_SIZE"]
    headroom_words = words * (constants["DISPATCH_PARENT_ZONE_MARKER_COUNT"] + constants["QUICK_PUSH_MARKER_COUNT"])
    headroom_words += constants["DISPATCH_META_DATA_UINT32_SIZE"] * constants["DISPATCH_META_DATA_COUNT"]
    assert DISPATCH_HEADROOM_MARKER_SLOTS * words == headroom_words
    push = kernel.split("void quick_push()", 1)[1].split("// Initiates a quick_push", 1)[0]
    assert "wIndex = CUSTOM_MARKERS;" in push
    assert "wIndex % NOC_ALIGNMENT_FACTOR" in push
    groups = ["pack", "instrn", "l1_1"]  # WH CLI schedules 89 records in one pass.
    payload_slots = 89 * 3 + 16
    assert counters_per_zone("wormhole_b0", groups) == 89
    assert single_pass_l1_headroom("wormhole_b0", groups) < 0
    # Two chunks repeat metadata, add two push markers, align independently,
    # and preserve the end-marker headroom checked by quick_push.
    expected_slots = 300
    assert expected_slots > payload_slots + 6
    assert bytes_per_program("wormhole_b0", groups) == expected_slots * 8
    support = recommend_program_support_count(20, safety=1, arch="wormhole_b0", groups=groups)
    assert support * PROGRAM_SUPPORT_BYTES >= 20 * expected_slots * 8


def test_alignment_and_impossible_stack_reserve_are_accounted_for():
    # Three counters + a 16-slot reserve + six metadata slots = 31 slots;
    # finish_profiler aligns this to 32, even without an intermediate flush.
    assert bytes_per_program("blackhole", ["fpu"]) == 32 * MARKER_BYTES
    with pytest.raises(ValueError, match="no safe"):  # allow-pytest.raises: no device conftest
        bytes_per_program("blackhole", ["fpu"], brisc_reserve=240)
