// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
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

/// Peak DRAM footprint over a tracking window, in bytes per device.
struct DramFootprint {
    uint64_t peak_allocated_bytes = 0;    ///< Highest usage reached.
    uint64_t min_largest_free_bytes = 0;  ///< Largest single buffer still allocatable at the tightest point.
};

/**
 * @brief Zero-overhead peak DRAM footprint tracking via the real device allocator.
 *
 * Unlike @ref MemoryUsageTracker (which captures the op graph, adding per-op overhead and modeling
 * an *estimated* peak), this reads the device allocator directly. While active, each DRAM allocation
 * samples the allocator on the allocation path itself -- no hooks, no capture buffers -- so it
 * measures the true peak DRAM footprint of the enclosed region at effectively zero cost, without
 * perturbing op timings.
 *
 * Reports a @ref DramFootprint: peak usage, and the min largest-free block (the contiguity that
 * limits allocation under fragmentation -- the first number to run out). Operates on the active
 * device from @ref ttml::autograd::ctx.
 */
namespace DramFootprintTracker {

/// Begin tracking; resets the footprint. Throws if already active. Not nestable, not thread safe.
void begin();

/// The footprint so far (running values), or zeros if not active. Not thread safe.
[[nodiscard]] DramFootprint footprint();

/// Stop tracking and return the final footprint, or zeros (with a warning) if not active. Not thread safe.
DramFootprint end();

}  // namespace DramFootprintTracker

/**
 * @brief RAII scope for @ref DramFootprintTracker: begins on construction, ends on destruction (so
 * tracking is always released, even on an exception). Read @ref footprint while the scope is alive:
 * @code
 *   ttml::utils::DramFootprintScope scope;
 *   run_training_step();
 *   auto peak = scope.footprint();
 * @endcode
 */
class DramFootprintScope {
public:
    DramFootprintScope();
    ~DramFootprintScope();
    DramFootprintScope(const DramFootprintScope&) = delete;
    DramFootprintScope& operator=(const DramFootprintScope&) = delete;
    DramFootprintScope(DramFootprintScope&&) = delete;
    DramFootprintScope& operator=(DramFootprintScope&&) = delete;

    /// The DRAM footprint so far within this scope, in bytes per device.
    [[nodiscard]] DramFootprint footprint() const;
};

/// DRAM reserved outside the allocator arena, in bytes per device (physical - arena). A static device
/// property (firmware base + any trace region), independent of any footprint tracking window.
[[nodiscard]] uint64_t dram_reserved_bytes();

/// DRAM arena (allocatable) size, in bytes per device -- the budget peak usage competes for. OOM is
/// gated by this, not by physical DRAM (reserved is never allocatable). physical = arena + reserved.
[[nodiscard]] uint64_t dram_arena_bytes();

}  // namespace ttml::utils
