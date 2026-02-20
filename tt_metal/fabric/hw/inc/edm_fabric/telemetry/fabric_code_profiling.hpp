// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "internal/ethernet/tt_eth_api.h"
#include "internal/risc_attribs.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/fabric_bandwidth_telemetry.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/telemetry/code_profiling_types.hpp"

#include <cstdint>
#include <cstddef>

/**
 * @brief Generic code profiling timer base class
 *
 * @tparam enabled Whether this timer is enabled (compile-time constant)
 * @tparam result_addr L1 address where results are stored
 */
template<bool enabled, size_t result_addr>
class CodeProfilingTimer {
private:
    RiscTimestamp start_ts;
    RiscTimestamp end_ts;
    bool should_dump = false;

public:
    // Intentionally don't initialize start_ts and end_ts here because they are only meaningful
    // after calling open and close. Initializing to default 0 values is just as garbage as
    // leaving them uninitialized.
    CodeProfilingTimer()=default;

    FORCE_INLINE void set_should_dump(bool dump) {
        should_dump = dump;
    }

    FORCE_INLINE void open() {
        // if (should_dump) {
        start_ts.full = eth_read_wall_clock();
        // }
    }

    FORCE_INLINE void close() {
        if (should_dump) {
            end_ts.full = eth_read_wall_clock();
            write_results();
        }
    }

private:
    FORCE_INLINE void write_results() {
        volatile tt_l1_ptr auto* result_ptr = reinterpret_cast<volatile tt_l1_ptr CodeProfilingTimerResult*>(result_addr);
        uint64_t duration = end_ts.full - start_ts.full;
        result_ptr->total_cycles += duration;
        result_ptr->num_instances += 1;
        if (duration < result_ptr->min_cycles) {
            result_ptr->min_cycles = duration;
        }
        if (duration > result_ptr->max_cycles) {
            result_ptr->max_cycles = duration;
        }
    }
};

/**
 * @brief Disabled timer specialization - completely empty
 */
template<size_t result_addr>
class CodeProfilingTimer<false, result_addr> {
public:
    CodeProfilingTimer() = default;
    FORCE_INLINE void set_should_dump(bool /*dump*/) {}
    FORCE_INLINE void open() {}
    FORCE_INLINE void close() {}
};

/**
 * @brief Helper function to get result address for a timer type
 * @tparam TimerType The timer type
 * @param buffer_base_addr The L1 buffer base address
 * @return The address where this timer's results are stored
 */
template<CodeProfilingTimerType TimerType>
constexpr size_t get_timer_result_addr(size_t buffer_base_addr) {
    return buffer_base_addr + (__builtin_ctz(static_cast<uint32_t>(TimerType)) * sizeof(CodeProfilingTimerResult));
}

/**
 * @brief Named profiler wrapper that simplifies timer usage
 *
 * @tparam TimerType The specific timer type (from CodeProfilingTimerType enum)
 * @tparam bitfield The enabled timers bitfield (compile-time constant)
 * @tparam buffer_base_addr The L1 buffer base address (compile-time constant)
 */
template<CodeProfilingTimerType TimerType, uint32_t bitfield, size_t buffer_base_addr>
class NamedProfiler {
private:
    static constexpr bool is_enabled = (bitfield & static_cast<uint32_t>(TimerType)) != 0;
    static constexpr size_t result_addr = get_timer_result_addr<TimerType>(buffer_base_addr);

    CodeProfilingTimer<is_enabled, result_addr> timer;

public:
    NamedProfiler() = default;

    FORCE_INLINE void set_should_dump(bool dump) {
        timer.set_should_dump(dump);
    }

    FORCE_INLINE void open() {
        timer.open();
    }

    FORCE_INLINE void close() {
        timer.close();
    }
};

/**
 * @brief Spin iteration counter for busy-wait loops.
 * Counts iterations inside spin loops and reports using the same L1 buffer format as timers.
 * total_cycles is repurposed as total iterations; num_instances = number of spin loop entries.
 *
 * @tparam CounterType The timer type enum value (determines L1 buffer slot)
 * @tparam bitfield The enabled timers bitfield (compile-time constant)
 * @tparam buffer_base_addr The L1 buffer base address (compile-time constant)
 */
template <CodeProfilingTimerType CounterType, uint32_t bitfield, size_t buffer_base_addr>
class SpinCounter {
    static constexpr bool is_enabled = (bitfield & static_cast<uint32_t>(CounterType)) != 0;
    static constexpr size_t result_addr = get_timer_result_addr<CounterType>(buffer_base_addr);
    uint32_t count = 0;

public:
    SpinCounter() = default;

    FORCE_INLINE void increment() {
        if constexpr (is_enabled) {
            count++;
        }
    }

    // Call after exiting the spin loop to write accumulated count to L1
    FORCE_INLINE void flush() {
        if constexpr (is_enabled) {
            volatile tt_l1_ptr auto* result_ptr =
                reinterpret_cast<volatile tt_l1_ptr CodeProfilingTimerResult*>(result_addr);
            result_ptr->total_cycles += count;
            result_ptr->num_instances += 1;
            if (count < result_ptr->min_cycles) {
                result_ptr->min_cycles = count;
            }
            if (count > result_ptr->max_cycles) {
                result_ptr->max_cycles = count;
            }
            count = 0;
        }
    }
};

/**
 * @brief Clear all code profiling results in the L1 buffer
 * @param buffer_base_addr The L1 buffer base address
 */
FORCE_INLINE void clear_code_profiling_buffer(size_t buffer_base_addr) {
    constexpr size_t num_timers = get_max_code_profiling_timer_types();
    volatile tt_l1_ptr auto* result_ptr =
        reinterpret_cast<volatile tt_l1_ptr CodeProfilingTimerResult*>(buffer_base_addr);

    for (size_t i = 0; i < num_timers; i++) {
        result_ptr[i].total_cycles = 0;
        result_ptr[i].num_instances = 0;
        result_ptr[i].min_cycles = 0xFFFFFFFFFFFFFFFFULL;  // UINT64_MAX sentinel
        result_ptr[i].max_cycles = 0;
    }
}
