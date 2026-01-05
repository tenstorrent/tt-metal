// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nlohmann/json.hpp>

#include "ttnn/common/guard.hpp"

namespace ttml::utils {

struct DRAMUsage {
    /** peak memory usage in bytes between begin_capture and end_capture */
    long long peak = 0;
    /** current memory usage in bytes at the time of end_capture */
    long long current = 0;
};

struct L1Usage {
    /** peak circular buffer usage in bytes between begin_capture and end_capture */
    long long peak_cb = 0;
    /** peak L1 buffer usage in bytes between begin_capture and end_capture */
    long long peak_buffer = 0;
    /** peak total (cb + buffer) usage in bytes between begin_capture and end_capture */
    long long peak_total = 0;
    /** current L1 buffer usage in bytes at the time of end_capture */
    long long current = 0;
};

namespace MemoryUsageTracker {
/**
 * @brief Begin capturing memory usage
 * @return A scope guard that will end the capture when it goes out of scope.
 * @warning Make sure to not ignore the return value of this function, otherwise trace will be empty!
 * @note This functions throws an exception if capture is already active.
 */
[[nodiscard]] ttnn::ScopeGuard begin_capture();

/**
 * @brief End capturing memory usage
 * @note If capture is not active, this function prints a warning
 */
void end_capture();

/**
 * @brief Get DRAM usage of captured trace
 * @note This function throws an exception if capture is not active
 * @return The DRAM usage
 */
DRAMUsage get_DRAM_usage();

/**
 * @brief Get L1 usage of captured trace
 * @note This function throws an exception if capture is not active
 * @return The L1 usage
 */
L1Usage get_L1_usage();

/**
 * @brief A convenience function to print memory usage
 */
void print_memory_usage();
}  // namespace MemoryUsageTracker
}  // namespace ttml::utils
