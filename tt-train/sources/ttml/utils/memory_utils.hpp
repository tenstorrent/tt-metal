// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nlohmann/json.hpp>

#include "ttnn/common/guard.hpp"
#include "ttnn/graph/graph_processor.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"

namespace ttml::utils {

using DRAMUsage = ttnn::graph::DRAMUsage;
using L1UsagePerCore = ttnn::graph::PeakMemoryUsagePerCore;

namespace MemoryUsageTracker {

inline constexpr const char* kDefaultTraceName = "END_TRACE";

/**
 * @brief Begin capturing memory usage
 * @return A scope guard that will end the capture when it goes out of scope.
 * @warning Make sure to not ignore the return value of this function, otherwise trace will be empty!
 * @note This functions throws an exception if capture is already active.
 * @note Not thread safe.
 */
[[nodiscard]] ttnn::ScopeGuard begin_capture(
    tt::tt_metal::IGraphProcessor::RunMode mode = tt::tt_metal::IGraphProcessor::RunMode::NORMAL);

/**
 * @brief End capturing memory usage and store the trace with the given name
 * @param name The name to store the trace under (default: "END_TRACE")
 * @note If capture is not active, this function prints a warning
 * @note Not thread safe.
 */
void end_capture(const std::string& name = kDefaultTraceName);

/**
 * @brief Create a checkpoint: save current trace with given name and start a new capture
 *
 * This function:
 * 1. Ends the current capture and saves the trace with the given name
 * 2. Starts a new capture session
 * 3. The new capture will carry forward the "current" usage from the checkpoint
 *    so that peak calculations in subsequent traces account for already-allocated memory
 *
 * @param name The name for this checkpoint
 * @note This function throws an exception if capture is not active
 * @note Not thread safe.
 */
void snapshot(const std::string& name);

/**
 * @brief Get DRAM usage of captured trace by name
 * @param name The name of the trace (default: "END_TRACE")
 * @note This function throws an exception if the named trace doesn't exist
 * @return The DRAM usage
 * @note Not thread safe.
 */
DRAMUsage get_dram_usage(const std::string& name = kDefaultTraceName);

/**
 * @brief Get DRAM usage of all captured traces
 * @return Vector of pairs of trace name and DRAM usage
 * @note Not thread safe.
 */
std::vector<std::pair<std::string, DRAMUsage>> get_dram_usage_all();

/**
 * @brief Get L1 usage of captured trace by name
 * @param name The name of the trace (default: "END_TRACE")
 * @note This function throws an exception if the named trace doesn't exist
 * @return The L1 usage
 * @note Not thread safe.
 */
L1UsagePerCore get_l1_usage(const std::string& name = kDefaultTraceName);

/**
 * @brief Get L1 usage of all captured traces
 * @return Vector of pairs of trace name and L1 usage
 * @note Not thread safe.
 */
std::vector<std::pair<std::string, L1UsagePerCore>> get_l1_usage_all();

/**
 * @brief Get all trace names in order they were captured
 * @return Vector of trace names
 * @note Not thread safe.
 */
std::vector<std::string> get_trace_names();

/**
 * @brief Print memory usage for all captured traces
 * @note Not thread safe.
 */
void print_memory_usage();

/**
 * @brief Clear all stored traces
 * @note Not thread safe.
 */
void clear();

}  // namespace MemoryUsageTracker
}  // namespace ttml::utils
