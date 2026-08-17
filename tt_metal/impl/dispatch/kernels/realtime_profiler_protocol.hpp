// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Completion counters use a nonzero sub-word width. Template parameters keep
// invalid shift counts from compiling in host tests or device firmware.
template <uint32_t CounterWidth>
constexpr uint32_t realtime_profiler_counter_mask() {
    static_assert(CounterWidth > 0 && CounterWidth < 32);
    return (1u << CounterWidth) - 1;
}

template <uint32_t CounterWidth>
constexpr uint32_t realtime_profiler_completion_target(uint32_t previous_count, uint32_t num_workers) {
    return (previous_count + num_workers) & realtime_profiler_counter_mask<CounterWidth>();
}

template <uint32_t CounterWidth>
constexpr bool realtime_profiler_stream_count_ge(uint32_t current, uint32_t target) {
    static_assert(CounterWidth > 0 && CounterWidth < 32);
    const uint32_t shifted_diff = (current - target) << (32 - CounterWidth);
    return static_cast<int32_t>(shifted_diff) >= 0;
}

constexpr bool realtime_profiler_queue_full(uint32_t write_index, uint32_t read_index, uint32_t capacity) {
    return write_index - read_index >= capacity;
}

constexpr bool realtime_profiler_generation_after(uint32_t generation, uint32_t adopted_generation) {
    const uint32_t distance = generation - adopted_generation;
    return distance != 0 && distance < (uint32_t{1} << 31);
}
