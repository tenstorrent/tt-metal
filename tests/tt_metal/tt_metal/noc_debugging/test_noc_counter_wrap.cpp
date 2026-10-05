// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <impl/debug/noc_debugging.hpp>

// The tracker treats a NOC transaction counter that moved backwards as a missing barrier. Those
// counters are carried in a 12-bit field (LocalNocEventDstTrailer), so they wrap every 4096
// transactions, and the comparison is a wrapping one -- rolling over must not be reported, while a
// counter that genuinely moved backwards must be.
//
// This drives that logic through push_event()/process_accumulated_events_all_chips() with synthetic
// events, so it needs no device and no kernel that issues 4096+ transactions. Doing it on device
// costs tens of seconds in a single dispatch.
namespace tt::tt_metal {
namespace {

constexpr size_t CHIP_ID = 0;
constexpr int PROCESSOR_ID = 0;
constexpr int OTHER_PROCESSOR_ID = 1;
constexpr int8_t CORE_X = 1;
constexpr int8_t CORE_Y = 1;

NocWriteEvent make_write(uint32_t src_addr, uint32_t counter_snapshot) {
    NocWriteEvent event{};
    event.src_addr = src_addr;
    event.dst_addr = 0x40000;
    event.num_bytes = 16;
    event.counter_snapshot = counter_snapshot;
    event.src_x = CORE_X;
    event.src_y = CORE_Y;
    event.dst_x = CORE_X;
    event.dst_y = CORE_Y;
    event.posted = false;
    event.noc = 0;
    event.is_semaphore = false;
    event.is_mcast = false;
    return event;
}

// Replays two writes whose counter snapshots go first -> second. The first only seeds the baseline;
// the second is the one compared against it. The two use different source addresses so that the
// separate "two writes from one source without a barrier" rule cannot account for the result.
bool flags_missing_barrier(uint32_t first_counter, uint32_t second_counter) {
    NOCDebugState state;
    state.push_event(CHIP_ID, /*timestamp=*/1, PROCESSOR_ID, make_write(0x1000, first_counter));
    state.push_event(CHIP_ID, /*timestamp=*/2, PROCESSOR_ID, make_write(0x2000, second_counter));
    state.process_accumulated_events_all_chips();

    const tt_cxy_pair core{CHIP_ID, {static_cast<size_t>(CORE_X), static_cast<size_t>(CORE_Y)}};
    return state.get_issues(core, PROCESSOR_ID).has_base_issue(NOCDebugIssueBaseType::WRITE_FLUSH_BARRIER);
}

TEST(NOCDebugCounterWrap, AdvancingCounterIsNotFlagged) { EXPECT_FALSE(flags_missing_barrier(100, 200)); }

TEST(NOCDebugCounterWrap, AdvanceAcrossTheWrapIsNotFlagged) {
    // 4090 -> 5 is an advance of 11 that happens to cross the 12-bit boundary, not a stall.
    EXPECT_FALSE(flags_missing_barrier(4090, 5));
}

TEST(NOCDebugCounterWrap, AdvanceFromTheLastValueBeforeWrapIsNotFlagged) {
    EXPECT_FALSE(flags_missing_barrier(4095, 0));
}

TEST(NOCDebugCounterWrap, StalledCounterIsFlagged) { EXPECT_TRUE(flags_missing_barrier(200, 100)); }

TEST(NOCDebugCounterWrap, ApparentJumpOverHalfTheCounterRangeIsFlagged) {
    EXPECT_TRUE(flags_missing_barrier(100, 4090));
}

TEST(NOCDebugAtomicTracking, BarrierOnlyDrainsIssuingProcessor) {
    NOCDebugState state;
    NocSemaphoreIncEvent atomic{};
    atomic.dst_addr = 0x40000;
    atomic.src_x = CORE_X;
    atomic.src_y = CORE_Y;
    atomic.dst_x = CORE_X;
    atomic.dst_y = CORE_Y;
    atomic.posted = false;
    atomic.noc = 0;
    atomic.is_semaphore = true;

    state.push_event(CHIP_ID, /*timestamp=*/1, PROCESSOR_ID, atomic);
    state.push_event(CHIP_ID, /*timestamp=*/2, OTHER_PROCESSOR_ID, atomic);
    state.push_event(
        CHIP_ID,
        /*timestamp=*/3,
        PROCESSOR_ID,
        NocAtomicBarrierEvent{.src_x = CORE_X, .src_y = CORE_Y, .noc = 0});
    state.process_accumulated_events_all_chips();
    state.finish_cores();

    const tt_cxy_pair core{CHIP_ID, {static_cast<size_t>(CORE_X), static_cast<size_t>(CORE_Y)}};
    EXPECT_FALSE(
        state.get_issues(core, PROCESSOR_ID).has_base_issue(NOCDebugIssueBaseType::UNFLUSHED_ATOMIC_AT_END));
    EXPECT_TRUE(
        state.get_issues(core, OTHER_PROCESSOR_ID).has_base_issue(NOCDebugIssueBaseType::UNFLUSHED_ATOMIC_AT_END));
}

}  // namespace
}  // namespace tt::tt_metal
