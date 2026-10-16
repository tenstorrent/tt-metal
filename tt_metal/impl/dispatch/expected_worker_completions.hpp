// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

#include <tt_stl/assert.hpp>
#include <tt-metalium/sub_device_types.hpp>

#include "impl/dispatch/dispatch_settings.hpp"

namespace tt::tt_metal {

// A dispatcher wait on a completion counter: the work it covers has completed once the counter reaches count.
struct CompletionWait {
    uint32_t counter = 0;
    uint32_t count = 0;
};

// A CQ's expected worker completion counts, one per completion counter, with the layout that maps sub-devices onto
// those counters: each sub-device owns num_completion_slots consecutive counters (DispatchSettings::
// num_completion_slots), starting at its base counter.
//
// Programs rotate through their sub-device's counters, and each program's go waits for the program before it on the
// sub-device. All of a sub-device's work has therefore completed once its current counter, the one its latest work
// reports into, reaches its count; control gos report into the current counter too.
//
// Sync counts that order a sub-device's work (the kernel config buffer manager, cross-node program ordering) are the
// sub-device's total expected completions across its counters. wait_for translates one into a counter wait.
struct ExpectedWorkerCompletions {
    // Programs kept per sub-device for translating sync counts into counter waits. A sync count older than every kept
    // program waits for the oldest one, which completes after it.
    static constexpr uint32_t SYNC_HISTORY_DEPTH = 16;

    // Where a program left its sub-device: the total expected completions and the wait that covers the program.
    struct SyncPoint {
        uint32_t total = 0;
        CompletionWait wait;
    };

    struct SubDeviceCompletions {
        // Expected completions across all of the sub-device's counters.
        uint32_t total = 0;
        // Slot, within the sub-device's block of counters, that its latest work reports into.
        uint32_t current_slot = 0;
        // Ring of the latest sync points, oldest at sync_history_start. Holds at least the point the sub-device's
        // counts were last cleared or loaded at.
        std::array<SyncPoint, SYNC_HISTORY_DEPTH> sync_history{};
        uint32_t sync_history_start = 0;
        uint32_t sync_history_size = 1;
    };

    // Where a program goes: its go waits for go_wait, and its workers report done into counter.
    struct ProgramCompletion {
        CompletionWait go_wait;
        uint32_t counter = 0;
    };

    CompletionCounterArray<uint32_t> counts{};
    DispatchArray<SubDeviceCompletions> sub_devices{};
    uint32_t num_sub_devices = 1;
    uint32_t num_completion_slots = 1;

    uint32_t& operator[](uint32_t counter_index) { return counts.at(counter_index); }
    uint32_t operator[](uint32_t counter_index) const { return counts.at(counter_index); }

    // First counter of sub_device_id's block. Traced programs report into it.
    uint32_t base_counter_index(SubDeviceId sub_device_id) const {
        TT_ASSERT(*sub_device_id < num_sub_devices);
        return DispatchSettings::completion_counter_index(*sub_device_id, num_completion_slots);
    }

    // Counter sub_device_id's latest work reports into. Once it reaches its count, all of the sub-device's work has
    // completed.
    uint32_t current_counter_index(SubDeviceId sub_device_id) const {
        const uint32_t index = base_counter_index(sub_device_id) + sub_devices[*sub_device_id].current_slot;
        TT_ASSERT(index < DispatchSettings::DISPATCH_COMPLETION_COUNTERS);
        return index;
    }

    // Wait for all of sub_device_id's work.
    CompletionWait current_wait(SubDeviceId sub_device_id) const {
        const uint32_t counter = current_counter_index(sub_device_id);
        return {counter, counts[counter]};
    }

    // Total expected completions across sub_device_id's counters, the sub-device's sync count.
    uint32_t total(SubDeviceId sub_device_id) const { return sub_devices[*sub_device_id].total; }

    // Places a program with num_workers workers on sub_device_id's next counter. A program without workers reports
    // nothing and stays on the current counter.
    ProgramCompletion add_program(SubDeviceId sub_device_id, uint32_t num_workers) {
        auto& sub_device = sub_devices[*sub_device_id];
        ProgramCompletion completion{
            .go_wait = current_wait(sub_device_id), .counter = current_counter_index(sub_device_id)};
        if (num_workers == 0) {
            return completion;
        }
        sub_device.current_slot = (sub_device.current_slot + 1) % num_completion_slots;
        completion.counter = current_counter_index(sub_device_id);
        counts[completion.counter] += num_workers;
        sub_device.total += num_workers;
        push_sync_point(sub_device, {sub_device.total, {completion.counter, counts[completion.counter]}});
        return completion;
    }

    // Wait that covers sub_device_id's work up to sync_count, a total the sub-device has reached.
    CompletionWait wait_for(SubDeviceId sub_device_id, uint32_t sync_count) const {
        const auto& sub_device = sub_devices[*sub_device_id];
        TT_ASSERT(sync_count <= sub_device.total, "Sync count {} is ahead of {}", sync_count, sub_device.total);
        for (uint32_t i = 0; i < sub_device.sync_history_size; ++i) {
            const SyncPoint& point = sub_device.sync_history[(sub_device.sync_history_start + i) % SYNC_HISTORY_DEPTH];
            if (point.total >= sync_count) {
                return point.wait;
            }
        }
        return current_wait(sub_device_id);
    }

    // Clears sub_device_id's counts, after its counters are cleared on the device.
    void clear(SubDeviceId sub_device_id) { load(sub_device_id, 0); }

    // Loads sub_device_id's counts after a trace whose programs reported num_completions into its base counter, with
    // its other counters cleared.
    void load(SubDeviceId sub_device_id, uint32_t num_completions) {
        const uint32_t base_counter = base_counter_index(sub_device_id);
        for (uint32_t slot = 0; slot < num_completion_slots; ++slot) {
            counts[base_counter + slot] = 0;
        }
        counts[base_counter] = num_completions;
        auto& sub_device = sub_devices[*sub_device_id];
        sub_device = SubDeviceCompletions{};
        sub_device.total = num_completions;
        sub_device.sync_history[0] = {num_completions, {base_counter, num_completions}};
    }

    // Clears every count, keeping the layout.
    void clear() {
        for (uint32_t i = 0; i < num_sub_devices; ++i) {
            clear(SubDeviceId{static_cast<uint8_t>(i)});
        }
    }

    // Clears every count and lays the counters out for num_sub_devices sub-devices.
    void reset(uint32_t num_sub_devices) {
        counts.fill(0);
        this->num_sub_devices = num_sub_devices;
        num_completion_slots = DispatchSettings::num_completion_slots(num_sub_devices);
        clear();
    }

private:
    static void push_sync_point(SubDeviceCompletions& sub_device, const SyncPoint& point) {
        const uint32_t end = (sub_device.sync_history_start + sub_device.sync_history_size) % SYNC_HISTORY_DEPTH;
        sub_device.sync_history[end] = point;
        if (sub_device.sync_history_size == SYNC_HISTORY_DEPTH) {
            sub_device.sync_history_start = (sub_device.sync_history_start + 1) % SYNC_HISTORY_DEPTH;
        } else {
            ++sub_device.sync_history_size;
        }
    }
};

}  // namespace tt::tt_metal
