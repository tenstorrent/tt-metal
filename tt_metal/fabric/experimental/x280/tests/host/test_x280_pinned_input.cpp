// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rt/x280_pinned_input.h"
#include "x280_pinned_input_record.hpp"

#include <gtest/gtest.h>
#include <hostdevcommon/fabric_common.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace {

// The routing image the pinned input holds is tt-metal's own routing table.
static_assert(X280_PINNED_INPUT_ROUTING_BYTES == sizeof(tt::tt_fabric::routing_l1_info_t));

TEST(X280PinnedInput, FixedRegionConstantsDescribeTheCompleteInputLayout) {
    EXPECT_EQ(X280_PINNED_INPUT_BASE, UINT64_C(0x0A000000));
    EXPECT_EQ(X280_PINNED_INPUT_SPAN, UINT64_C(0x10C0));
    EXPECT_EQ(X280_PINNED_INPUT_END, UINT64_C(0x0A0010C0));
    EXPECT_EQ(X280_PINNED_INPUT_CACHE_LINE_BYTES, 64u);
    EXPECT_EQ(X280_PINNED_INPUT_CACHE_LINES, 67u);
    EXPECT_EQ(X280_PINNED_INPUT_CACHE_LINES * X280_PINNED_INPUT_CACHE_LINE_BYTES, X280_PINNED_INPUT_SPAN);

    EXPECT_EQ(X280_PINNED_INPUT_ROUTING_OFFSET, 0u);
    EXPECT_EQ(X280_PINNED_INPUT_ROUTING_BYTES, 2704u);
    EXPECT_EQ(X280_PINNED_INPUT_HEADERS_OFFSET, UINT64_C(0xAC0));
    EXPECT_EQ(X280_PINNED_INPUT_HEADER_COUNT, 16u);
    EXPECT_EQ(X280_PINNED_INPUT_HEADER_BYTES, 96u);
    EXPECT_EQ(X280_PINNED_INPUT_HEADERS_BYTES, X280_PINNED_INPUT_HEADER_COUNT * X280_PINNED_INPUT_HEADER_BYTES);
    EXPECT_LE(X280_PINNED_INPUT_ROUTING_OFFSET + X280_PINNED_INPUT_ROUTING_BYTES, X280_PINNED_INPUT_HEADERS_OFFSET);
    EXPECT_EQ(X280_PINNED_INPUT_HEADERS_OFFSET + X280_PINNED_INPUT_HEADERS_BYTES, X280_PINNED_INPUT_SPAN);
    EXPECT_EQ(X280_PINNED_INPUT_FIRST_HEADER_BASE, UINT64_C(0x0A000AC0));
    EXPECT_EQ(X280_PINNED_INPUT_FIRST_HEADER_BASE % X280_PINNED_INPUT_CACHE_LINE_BYTES, 0u);
    EXPECT_NE(
        x280_pinned_input_contains(
            &X280_PINNED_INPUT_ROUTER_REGION, X280_PINNED_INPUT_FIRST_HEADER_BASE, X280_PINNED_INPUT_HEADER_BYTES),
        0);
}

TEST(X280PinnedInput, ContainsMatchesEveryLocalHalfOpenBoundarySpan) {
    const uint64_t first_address = X280_PINNED_INPUT_BASE - 1u;
    const uint64_t last_address = X280_PINNED_INPUT_END + 1u;
    const uint64_t largest_local_span = X280_PINNED_INPUT_SPAN + 1u;

    for (uint64_t address = first_address; address <= last_address; ++address) {
        for (uint64_t bytes = 0; bytes <= largest_local_span; ++bytes) {
            const bool expected = address >= X280_PINNED_INPUT_BASE && address <= X280_PINNED_INPUT_END &&
                                  bytes <= X280_PINNED_INPUT_END - address;
            const bool actual = x280_pinned_input_contains(&X280_PINNED_INPUT_ROUTER_REGION, address, bytes) != 0;
            if (actual != expected) {
                FAIL() << "address=" << address << " bytes=" << bytes << " expected=" << expected
                       << " actual=" << actual;
            }
        }
    }
}

TEST(X280PinnedInput, ContainsRejectsRemoteAndOverflowingRanges) {
    struct Range {
        uint64_t address;
        uint64_t bytes;
    };
    constexpr std::array<Range, 10> rejected = {{
        {X280_PINNED_INPUT_BASE - 1u, 0u},
        {X280_PINNED_INPUT_BASE - 1u, 1u},
        {X280_PINNED_INPUT_END + 1u, 0u},
        {X280_PINNED_INPUT_END + 1u, 1u},
        {X280_PINNED_INPUT_BASE, std::numeric_limits<uint64_t>::max()},
        {X280_PINNED_INPUT_BASE, std::numeric_limits<uint64_t>::max() - X280_PINNED_INPUT_BASE},
        {X280_PINNED_INPUT_BASE, std::numeric_limits<uint64_t>::max() - X280_PINNED_INPUT_BASE + 1u},
        {X280_PINNED_INPUT_END, std::numeric_limits<uint64_t>::max()},
        {std::numeric_limits<uint64_t>::max() - 1u, 2u},
        {std::numeric_limits<uint64_t>::max(), 1u},
    }};

    for (const auto& range : rejected) {
        EXPECT_EQ(x280_pinned_input_contains(&X280_PINNED_INPUT_ROUTER_REGION, range.address, range.bytes), 0)
            << "address=" << range.address << " bytes=" << range.bytes;
    }
}

TEST(X280PinnedInput, WritableSamplesRegionValidation) {
    // Valid writable samples descriptor
    x280_pinned_input_region_t valid_region{
        .base = X280_PINNED_INPUT_BASE,
        .bytes = 107520u,  // multiple of 64 <= 131072
        .kind = static_cast<uint32_t>(X280_PINNED_INPUT_WRITABLE_SAMPLES),
        .reserved = 0,
        .reserved_tail = 0,
    };
    EXPECT_NE(x280_pinned_input_validate_region(&valid_region), 0);
    EXPECT_NE(x280_pinned_input_contains(&valid_region, X280_PINNED_INPUT_BASE, 107520u), 0);
    EXPECT_EQ(x280_pinned_input_contains(&valid_region, X280_PINNED_INPUT_BASE, 107521u), 0);

    // Full 128-KiB capacity
    valid_region.bytes = 131072u;
    EXPECT_NE(x280_pinned_input_validate_region(&valid_region), 0);

    // Excess capacity > 131072
    x280_pinned_input_region_t excess_region = valid_region;
    excess_region.bytes = 131072u + 64u;
    EXPECT_EQ(x280_pinned_input_validate_region(&excess_region), 0);

    // Not multiple of 64
    x280_pinned_input_region_t unaligned_region = valid_region;
    unaligned_region.bytes = 1024u + 1u;
    EXPECT_EQ(x280_pinned_input_validate_region(&unaligned_region), 0);

    // Wrong base
    x280_pinned_input_region_t wrong_base = valid_region;
    wrong_base.base = 0x08020000u;
    EXPECT_EQ(x280_pinned_input_validate_region(&wrong_base), 0);

    // Nonzero reserved
    x280_pinned_input_region_t nonzero_reserved = valid_region;
    nonzero_reserved.reserved = 1u;
    EXPECT_EQ(x280_pinned_input_validate_region(&nonzero_reserved), 0);
}

TEST(X280PinnedInput, PreparedPolicyConstantsKeepOneGeneralPurposeWay) {
    EXPECT_EQ(X280_PINNED_INPUT_WAYENABLE, 1u);
    EXPECT_EQ(X280_PINNED_INPUT_GENERAL_WAY, 0u);
    EXPECT_EQ(X280_PINNED_INPUT_RESERVED_WAY, 1u);
    EXPECT_EQ(X280_PINNED_INPUT_ACTIVE_BITS, UINT64_C(0x3));
    EXPECT_EQ(X280_PINNED_INPUT_BASELINE_BITS, UINT64_C(0x3));
    EXPECT_EQ(X280_PINNED_INPUT_PREPARED_BITS, UINT64_C(0x1));
    EXPECT_EQ(X280_PINNED_INPUT_MASTER_COUNT, 38u);
    EXPECT_EQ(X280_PINNED_INPUT_POPULATION_MASTER, 34u);
}

TEST(X280PinnedInput, AdmissionRecordHasTheNativeCheckWireLayout) {
    EXPECT_EQ(X280_PINNED_INPUT_ADMITTED, 0);
    EXPECT_EQ(X280_PINNED_INPUT_WRONG_WAYENABLE, 1);
    EXPECT_EQ(X280_PINNED_INPUT_WRONG_WAYMASK, 2);
    EXPECT_EQ(sizeof(x280_pinned_input_admission_t), 16u);
    EXPECT_EQ(alignof(x280_pinned_input_admission_t), alignof(uint64_t));
    EXPECT_EQ(offsetof(x280_pinned_input_admission_t, code), 0u);
    EXPECT_EQ(offsetof(x280_pinned_input_admission_t, master_id), 4u);
    EXPECT_EQ(offsetof(x280_pinned_input_admission_t, observed), 8u);
}

using x280::PinnedInputPrepareOutcome;
using x280::PinnedInputStatus;

x280::PinnedInputState state_with(PinnedInputStatus status) {
    x280::PinnedInputState state;
    state.status = status;
    return state;
}

TEST(X280PinnedInputRecord, DecodesEachState) {
    const auto prepared =
        x280::decode_pinned_input_record(x280::encode_pinned_input_record(PinnedInputStatus::Prepared));
    EXPECT_EQ(prepared.status, PinnedInputStatus::Prepared);
    EXPECT_EQ(prepared.wayenable, X280_PINNED_INPUT_WAYENABLE);
    EXPECT_EQ(prepared.region_base, X280_PINNED_INPUT_BASE);
    EXPECT_EQ(prepared.region_bytes, X280_PINNED_INPUT_SPAN);
    EXPECT_EQ(
        x280::decode_pinned_input_record(x280::encode_pinned_input_record(PinnedInputStatus::Aborted)).status,
        PinnedInputStatus::Aborted);

    x280::PinnedInputRecord words{};
    EXPECT_EQ(x280::decode_pinned_input_record(words).status, PinnedInputStatus::None);
    words.fill(0xFFFFFFFFu);
    EXPECT_EQ(x280::decode_pinned_input_record(words).status, PinnedInputStatus::InvalidObservation);

    struct Case {
        const char* what;
        size_t word;
        uint32_t value;
    };
    for (const Case& c : {
             Case{"wrong magic", x280::kPinnedInputWordMagic, x280::kPinnedInputRecordMagic + 1},
             Case{"unknown status", x280::kPinnedInputWordStatus, 3},
             Case{"reserved word", x280::kPinnedInputRecordWords - 1, 1},
             Case{"one all-ones word", x280::kPinnedInputWordBytesHi, 0xFFFFFFFFu},
         }) {
        words = x280::encode_pinned_input_record(PinnedInputStatus::Prepared);
        words[c.word] = c.value;
        EXPECT_EQ(x280::decode_pinned_input_record(words).status, PinnedInputStatus::Corrupt) << c.what;
    }
}

TEST(X280PinnedInputRecord, PrepareRefusesInOrder) {
    const auto fixed = x280::decode_pinned_input_record(x280::encode_pinned_input_record(PinnedInputStatus::Prepared));
    auto other_span = fixed;
    other_span.region_bytes += X280_PINNED_INPUT_CACHE_LINE_BYTES;
    const auto none = state_with(PinnedInputStatus::None);
    const auto aborted = state_with(PinnedInputStatus::Aborted);
    constexpr uint8_t ways = x280::kPreparedLimWays;
    constexpr uint32_t wayenable = X280_PINNED_INPUT_WAYENABLE;

    struct Case {
        const char* what;
        uint8_t lim_ways;
        uint32_t raw_wayenable;
        bool busy;
        x280::PinnedInputState state;
        std::optional<PinnedInputPrepareOutcome> expected;
    };
    for (const Case& c : {
             Case{"proceeds from none", ways, wayenable, false, none, std::nullopt},
             Case{
                 "all-ones WayEnable", ways, 0xFFFFFFFFu, true, aborted, PinnedInputPrepareOutcome::InvalidObservation},
             Case{
                 "runtime at the reset shape",
                 x280::kMaxLimWays,
                 wayenable,
                 true,
                 aborted,
                 PinnedInputPrepareOutcome::WrongShape},
             Case{"tile at the reset shape", ways, 0, true, aborted, PinnedInputPrepareOutcome::WrongShape},
             Case{"live session", ways, wayenable, true, fixed, PinnedInputPrepareOutcome::Busy},
             Case{"already prepared", ways, wayenable, false, fixed, PinnedInputPrepareOutcome::AlreadyPrepared},
             Case{
                 "prepared at another span",
                 ways,
                 wayenable,
                 false,
                 other_span,
                 PinnedInputPrepareOutcome::ResetRequired},
             Case{"aborted", ways, wayenable, false, aborted, PinnedInputPrepareOutcome::ResetRequired},
             Case{
                 "unreadable record",
                 ways,
                 wayenable,
                 false,
                 state_with(PinnedInputStatus::InvalidObservation),
                 PinnedInputPrepareOutcome::ResetRequired},
         }) {
        EXPECT_EQ(x280::refuse_pinned_input_prepare(c.lim_ways, c.raw_wayenable, c.busy, c.state), c.expected)
            << c.what;
    }
}

x280::PinnedInputPrepareResult successful_run() {
    x280::PinnedInputPrepareResult result;
    auto& enqueue = result.enqueue.emplace();
    enqueue.outcome = x280::EnqueueOutcome::Enqueued;
    enqueue.session.value = 1;
    auto& wait = result.wait.emplace();
    wait.outcome = x280::WaitOutcome::ResultReady;
    for (uint32_t hart = 0; hart < x280::kHartsPerTile; ++hart) {
        wait.hartstates[hart].observation_valid = true;
        wait.hartstates[hart].state.result =
            hart == 0 ? X280_PINNED_INPUT_RESULT_PREPARED : X280_PINNED_INPUT_RESULT_NOOP_HART;
    }
    result.cleanup.emplace().outcome = x280::CleanupOutcome::ReturnedToIdle;
    auto& final_state = result.final_state.emplace();
    final_state.observation_valid = true;
    final_state.resident_idle_live = true;
    return result;
}

TEST(X280PinnedInputRecord, OnlyAConfirmedRunCommitsAndAnyLaterFailureLeavesAborted) {
    EXPECT_EQ(x280::pinned_input_status_after_run(successful_run()), PinnedInputStatus::Prepared);

    auto never_ran = successful_run();
    never_ran.enqueue->outcome = x280::EnqueueOutcome::ImageRejected;
    never_ran.enqueue->session = {};
    EXPECT_EQ(x280::pinned_input_status_after_run(never_ran), PinnedInputStatus::None);

    struct Case {
        const char* what;
        void (*mutate)(x280::PinnedInputPrepareResult&);
    };
    for (const Case& c : {
             Case{"not submitted", [](auto& r) { r.enqueue.reset(); }},
             Case{
                 "JUMP uncertain",
                 [](auto& r) { r.enqueue->outcome = x280::EnqueueOutcome::JumpPublicationUncertain; }},
             Case{"wait timed out", [](auto& r) { r.wait->outcome = x280::WaitOutcome::TimedOut; }},
             Case{
                 "hart 0 tore down",
                 [](auto& r) { r.wait->hartstates[0].state.result = X280_PINNED_INPUT_RESULT_TORN_DOWN; }},
             Case{
                 "hart 3 not a no-op",
                 [](auto& r) { r.wait->hartstates[3].state.result = X280_PINNED_INPUT_RESULT_PREPARED; }},
             Case{"hart 1 unreadable", [](auto& r) { r.wait->hartstates[1].observation_valid = false; }},
             Case{"cleanup failed", [](auto& r) { r.cleanup->outcome = x280::CleanupOutcome::StillRunningActiveFw; }},
             Case{"not back in resident idle", [](auto& r) { r.final_state->resident_idle_live = false; }},
         }) {
        auto result = successful_run();
        c.mutate(result);
        EXPECT_EQ(x280::pinned_input_status_after_run(result), PinnedInputStatus::Aborted) << c.what;
    }
}

}  // namespace
