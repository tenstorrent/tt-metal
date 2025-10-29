// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <bit>
/**
 * @brief Enumeration of code profiling timer types as bitfield
 * Each timer type is a unique bit position, allowing multiple timers to be enabled simultaneously
 */
enum class CodeProfilingTimerType : uint32_t {
    NONE = 0,
    RECEIVER_CHANNEL_FORWARD = 1 << 0,
    // Future timers can be added here:
    // SENDER_CHANNEL_PROCESS = 1 << 1,
    // PACKET_ROUTING = 1 << 2,
    // etc.
    LAST = RECEIVER_CHANNEL_FORWARD << 1  // Sentinel for size calculation
};

/**
 * @brief Structure to store accumulated code profiling results for a single timer type
 */
struct CodeProfilingTimerResult {
    uint64_t total_cycles;    // Total cycles accumulated across all captures
    uint64_t num_instances;   // Number of timer captures that occurred
};

/**
 * @brief Get the number of timer types defined in the enum
 * @return Number of timer types (excluding NONE and LAST)
 */
constexpr uint32_t get_num_code_profiling_timer_types() {
    // Count the number of timer types defined
    // Currently only RECEIVER_CHANNEL_FORWARD is defined
    return 1;
}

/**
 * @brief Get the maximum number of timer types supported
 * @return Maximum number of timer types
 */
constexpr uint32_t get_max_code_profiling_timer_types() {
    // get the bit offset of LAST
    return __builtin_ctz(static_cast<uint32_t>(CodeProfilingTimerType::LAST));
}
