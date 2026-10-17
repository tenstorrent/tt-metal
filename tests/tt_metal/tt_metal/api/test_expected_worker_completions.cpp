// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "impl/dispatch/expected_worker_completions.hpp"

using namespace tt::tt_metal;

namespace expected_worker_completions_tests {

constexpr SubDeviceId kSubDevice0{0};
constexpr SubDeviceId kSubDevice1{1};

// Programs rotate through their sub-device's counters, and each program's go waits for the program before it.
TEST(ExpectedWorkerCompletions, CPU_ProgramsRotateThroughSubDeviceCounters) {
    ExpectedWorkerCompletions completions;
    completions.reset(2);
    ASSERT_EQ(completions.num_completion_slots, 4u);
    ASSERT_EQ(completions.base_counter_index(kSubDevice1), 4u);

    CompletionWait previous = completions.current_wait(kSubDevice1);
    EXPECT_EQ(previous.counter, 4u);
    EXPECT_EQ(previous.count, 0u);
    for (uint32_t i = 0; i < 9; ++i) {
        const auto completion = completions.add_program(kSubDevice1, 3);
        const uint32_t expected_counter = 4 + ((i + 1) % 4);
        EXPECT_EQ(completion.counter, expected_counter) << i;
        EXPECT_EQ(completion.go_wait.counter, previous.counter) << i;
        EXPECT_EQ(completion.go_wait.count, previous.count) << i;
        // Programs i, i - 4, ... report into the counter.
        EXPECT_EQ(completions[expected_counter], 3 * ((i / 4) + 1)) << i;
        EXPECT_EQ(completions.current_counter_index(kSubDevice1), expected_counter) << i;
        EXPECT_EQ(completions.total(kSubDevice1), 3 * (i + 1)) << i;
        previous = completions.current_wait(kSubDevice1);
    }
    // The other sub-device's counters are untouched.
    EXPECT_EQ(completions.current_wait(kSubDevice0).counter, 0u);
    EXPECT_EQ(completions.total(kSubDevice0), 0u);
    for (uint32_t counter = 0; counter < 4; ++counter) {
        EXPECT_EQ(completions[counter], 0u);
    }
}

// A program without workers reports nothing, so it stays on the current counter.
TEST(ExpectedWorkerCompletions, CPU_ProgramWithoutWorkersStaysOnCurrentCounter) {
    ExpectedWorkerCompletions completions;
    completions.reset(1);
    completions.add_program(kSubDevice0, 5);
    const CompletionWait before = completions.current_wait(kSubDevice0);
    const auto completion = completions.add_program(kSubDevice0, 0);
    EXPECT_EQ(completion.counter, before.counter);
    EXPECT_EQ(completion.go_wait.counter, before.counter);
    EXPECT_EQ(completion.go_wait.count, before.count);
    EXPECT_EQ(completions.current_wait(kSubDevice0).counter, before.counter);
    EXPECT_EQ(completions.total(kSubDevice0), 5u);
}

// With as many sub-devices as counters, each sub-device has one counter and every program reports into it.
TEST(ExpectedWorkerCompletions, CPU_OneSlotPerSubDeviceDoesNotRotate) {
    ExpectedWorkerCompletions completions;
    completions.reset(DispatchSettings::DISPATCH_COMPLETION_COUNTERS);
    ASSERT_EQ(completions.num_completion_slots, 1u);
    for (uint32_t i = 0; i < 3; ++i) {
        const auto completion = completions.add_program(kSubDevice1, 2);
        EXPECT_EQ(completion.counter, 1u);
        EXPECT_EQ(completion.go_wait.counter, 1u);
        EXPECT_EQ(completion.go_wait.count, 2 * i);
    }
}

// A sync count translates into the wait for the first program that reached it.
TEST(ExpectedWorkerCompletions, CPU_SyncCountWaitsForProgramThatReachedIt) {
    ExpectedWorkerCompletions completions;
    completions.reset(2);
    std::vector<CompletionWait> program_waits;
    std::vector<uint32_t> totals;
    for (uint32_t i = 0; i < 6; ++i) {
        completions.add_program(kSubDevice0, i + 1);
        program_waits.push_back(completions.current_wait(kSubDevice0));
        totals.push_back(completions.total(kSubDevice0));
    }
    for (size_t i = 0; i < program_waits.size(); ++i) {
        const CompletionWait wait = completions.wait_for(kSubDevice0, totals[i]);
        EXPECT_EQ(wait.counter, program_waits[i].counter) << i;
        EXPECT_EQ(wait.count, program_waits[i].count) << i;
        // A sync count between two programs' totals waits for the later program.
        if (i > 0) {
            const CompletionWait between = completions.wait_for(kSubDevice0, totals[i - 1] + 1);
            EXPECT_EQ(between.counter, wait.counter) << i;
            EXPECT_EQ(between.count, wait.count) << i;
        }
    }
    // Sync count 0 needs nothing beyond the cleared counts.
    const CompletionWait start = completions.wait_for(kSubDevice0, 0);
    EXPECT_EQ(start.counter, completions.base_counter_index(kSubDevice0));
    EXPECT_EQ(start.count, 0u);
}

// A sync count older than every kept program waits for the oldest kept program, which completes after it.
TEST(ExpectedWorkerCompletions, CPU_OldSyncCountWaitsForOldestKeptProgram) {
    ExpectedWorkerCompletions completions;
    completions.reset(1);
    constexpr uint32_t kNumPrograms = ExpectedWorkerCompletions::SYNC_HISTORY_DEPTH + 5;
    std::vector<CompletionWait> program_waits;
    for (uint32_t i = 0; i < kNumPrograms; ++i) {
        completions.add_program(kSubDevice0, 1);
        program_waits.push_back(completions.current_wait(kSubDevice0));
    }
    const uint32_t oldest_kept = kNumPrograms - ExpectedWorkerCompletions::SYNC_HISTORY_DEPTH;
    for (uint32_t sync_count = 0; sync_count <= oldest_kept + 1; ++sync_count) {
        const CompletionWait wait = completions.wait_for(kSubDevice0, sync_count);
        // Program i reaches total i + 1.
        const uint32_t program = std::max(sync_count, oldest_kept + 1) - 1;
        EXPECT_EQ(wait.counter, program_waits[program].counter) << sync_count;
        EXPECT_EQ(wait.count, program_waits[program].count) << sync_count;
    }
}

// Loading a trace's completions puts them on the base counter and clears the sub-device's other counters.
TEST(ExpectedWorkerCompletions, CPU_LoadPutsTraceCompletionsOnBaseCounter) {
    ExpectedWorkerCompletions completions;
    completions.reset(2);
    for (uint32_t i = 0; i < 3; ++i) {
        completions.add_program(kSubDevice0, 4);
    }
    completions.add_program(kSubDevice1, 4);
    completions.load(kSubDevice0, 10);
    EXPECT_EQ(completions[0], 10u);
    for (uint32_t counter = 1; counter < 4; ++counter) {
        EXPECT_EQ(completions[counter], 0u) << counter;
    }
    EXPECT_EQ(completions.current_counter_index(kSubDevice0), 0u);
    EXPECT_EQ(completions.total(kSubDevice0), 10u);
    const CompletionWait wait = completions.wait_for(kSubDevice0, 10);
    EXPECT_EQ(wait.counter, 0u);
    EXPECT_EQ(wait.count, 10u);
    // The next program moves on from the base counter.
    EXPECT_EQ(completions.add_program(kSubDevice0, 1).counter, 1u);
    // The other sub-device is untouched.
    EXPECT_EQ(completions.current_counter_index(kSubDevice1), 5u);
    EXPECT_EQ(completions[5], 4u);
}

}  // namespace expected_worker_completions_tests
