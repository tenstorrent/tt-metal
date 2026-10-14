// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt_stl/assert.hpp>
#include <tt-metalium/sub_device_types.hpp>

#include "impl/dispatch/dispatch_settings.hpp"

namespace tt::tt_metal {

// A CQ's expected worker completion counts, one per completion counter, with the layout that maps sub-devices onto
// those counters: each sub-device owns num_completion_slots consecutive counters (DispatchSettings::
// num_completion_slots), starting at its base counter.
struct ExpectedWorkerCompletions {
    CompletionCounterArray<uint32_t> counts{};
    uint32_t num_completion_slots = 1;

    // Base completion counter of sub_device_id, the one its programs and control acks report done into.
    uint32_t counter_index(SubDeviceId sub_device_id) const {
        const uint32_t index = DispatchSettings::completion_counter_index(*sub_device_id, num_completion_slots);
        TT_ASSERT(index < DispatchSettings::DISPATCH_COMPLETION_COUNTERS);
        return index;
    }

    uint32_t& operator[](uint32_t counter_index) { return counts.at(counter_index); }
    uint32_t operator[](uint32_t counter_index) const { return counts.at(counter_index); }

    // Clears every count and lays the counters out for num_sub_devices sub-devices.
    void reset(uint32_t num_sub_devices) {
        counts.fill(0);
        num_completion_slots = DispatchSettings::num_completion_slots(num_sub_devices);
    }
};

}  // namespace tt::tt_metal
