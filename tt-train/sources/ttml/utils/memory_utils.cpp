// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "memory_utils.hpp"

#include <algorithm>
#include <ranges>
#include <string_view>

namespace ttml::utils {
using namespace ttnn::graph;

namespace MemoryUsageTracker {
namespace {

constexpr double KB = 1024.0;
constexpr double MB = KB * KB;

using NamedTrace = std::pair<std::string, nlohmann::json>;

// Global variables for memory usage tracking
bool is_capture_active = false;
std::shared_ptr<ttnn::graph::GraphProcessor> graph_processor;
tt::tt_metal::IGraphProcessor::RunMode capture_mode = tt::tt_metal::IGraphProcessor::RunMode::NORMAL;

std::vector<NamedTrace> traces;

// Returns the first segment recorded under `name`, or throws if there is none.
const nlohmann::json& find_trace(std::string_view name) {
    if (auto it = std::ranges::find(traces, name, &NamedTrace::first); it != traces.end()) {
        return it->second;
    }
    throw std::runtime_error(fmt::format("MemoryUsageTracker: Trace '{}' not found", name));
}

}  // namespace

ttnn::ScopeGuard begin_capture(tt::tt_metal::IGraphProcessor::RunMode mode) {
    if (is_capture_active) {
        throw std::runtime_error("MemoryUsageTracker: Capture already active");
    }
    capture_mode = mode;
    graph_processor = std::make_shared<ttnn::graph::GraphProcessor>(mode);
    graph_processor->begin_graph_capture(mode);
    is_capture_active = true;
    return ttnn::make_guard([]() {
        end_capture();
        clear();
    });
}

void end_capture(const std::string& name) {
    if (is_capture_active) {
        traces.emplace_back(name, graph_processor->end_graph_capture());
        is_capture_active = false;
    }
}

void snapshot(const std::string& name) {
    if (!is_capture_active) {
        throw std::runtime_error("MemoryUsageTracker: Cannot snapshot - capture is not active");
    }

    traces.emplace_back(name, graph_processor->end_graph_capture());

    // Start a new capture
    graph_processor = std::make_shared<ttnn::graph::GraphProcessor>(capture_mode);
    graph_processor->begin_graph_capture(capture_mode);
}

DRAMUsage get_dram_usage(const std::string& name) {
    return ttnn::graph::extract_dram_usage(find_trace(name));
}

std::vector<std::pair<std::string, DRAMUsage>> get_dram_usage_all() {
    std::vector<std::pair<std::string, DRAMUsage>> result;
    result.reserve(traces.size());
    for (const auto& [name, trace] : traces) {
        result.emplace_back(name, ttnn::graph::extract_dram_usage(trace));
    }
    return result;
}

L1UsagePerCore get_l1_usage(const std::string& name) {
    return ttnn::graph::extract_resource_usage_per_core(find_trace(name));
}

std::vector<std::pair<std::string, L1UsagePerCore>> get_l1_usage_all() {
    std::vector<std::pair<std::string, L1UsagePerCore>> result;
    result.reserve(traces.size());
    for (const auto& [name, trace] : traces) {
        result.emplace_back(name, ttnn::graph::extract_resource_usage_per_core(trace));
    }
    return result;
}

std::vector<std::string> get_trace_names() {
    auto names = traces | std::views::keys;
    return {names.begin(), names.end()};
}

void print_memory_usage() {
    if (traces.empty()) {
        fmt::print("WARNING: No traces captured\n");
        return;
    }

    fmt::print("=== Memory Usage Summary ===\n\n");

    // Track cumulative values across checkpoints
    // cumulative_current = sum of all changes up to this point (actual memory in use)
    // cumulative_peak = max(prev_cumulative_peak, prev_cumulative_current + this_segment_peak)
    long long cumulative_current = 0;
    long long cumulative_peak = 0;

    for (const auto& [name, trace] : traces) {
        auto dram_usage = ttnn::graph::extract_dram_usage(trace);
        auto l1_usage = ttnn::graph::extract_resource_usage_per_core(trace);

        // Calculate cumulative values for this checkpoint
        // The segment's peak is relative to start of segment, so real peak during this segment
        // is previous_current + segment_peak
        long long segment_absolute_peak = cumulative_current + dram_usage.peak;
        cumulative_peak = std::max(cumulative_peak, segment_absolute_peak);
        cumulative_current += dram_usage.total_allocations - dram_usage.total_deallocations;

        fmt::print("--- {} ---\n", name);
        fmt::print(
            "  DRAM: Segment Peak {:.2f} MB, Allocations {:.2f} MB, Deallocations {:.2f} MB, Segment Change {:+.2f} "
            "MB\n",
            dram_usage.peak / MB,
            dram_usage.total_allocations / MB,
            dram_usage.total_deallocations / MB,
            dram_usage.total_allocations / MB - dram_usage.total_deallocations / MB);
        fmt::print(
            "  DRAM: Cumulative Peak {:.2f} MB, Cumulative Current {:.2f} MB\n",
            cumulative_peak / MB,
            cumulative_current / MB);
        fmt::print(
            "  L1: Peak CB {:.2f} MB, Peak Buffer {:.2f} MB, Peak Total {:.2f} MB\n",
            l1_usage.peak_cb / MB,
            l1_usage.peak_l1 / MB,
            l1_usage.peak_total / MB);
    }

    fmt::print("\n=== Final Totals ===\n");
    fmt::print(
        "Overall DRAM Peak: {:.2f} MB, Final DRAM Usage: {:.2f} MB\n", cumulative_peak / MB, cumulative_current / MB);
    fmt::print("\n");
}

void clear() {
    traces.clear();
    is_capture_active = false;
    graph_processor.reset();
}

}  // namespace MemoryUsageTracker

}  // namespace ttml::utils
