// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

#include "ttnn/common/guard.hpp"

namespace ttml::utils {

struct DRAMUsage {
    /** device id -> peak memory usage in bytes between begin_capture and end_capture */
    std::unordered_map<std::string, long long> peak;
    /** device id -> current memory usage in bytes at the time of end_capture */
    std::unordered_map<std::string, long long> current;
};

struct L1Usage {
    /** device id -> peak circular buffer usage in bytes between begin_capture and end_capture */
    std::unordered_map<std::string, long long> peak_cb;
    /** device id -> peak L1 buffer usage in bytes between begin_capture and end_capture */
    std::unordered_map<std::string, long long> peak_buffer;
    /** device id -> peak total (cb + buffer) usage in bytes between begin_capture and end_capture */
    std::unordered_map<std::string, long long> peak_total;
    /** device id -> current L1 buffer usage in bytes at the time of end_capture */
    std::unordered_map<std::string, long long> current;
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
