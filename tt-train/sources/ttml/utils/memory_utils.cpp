// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "memory_utils.hpp"

#include <fmt/format.h>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer_types.hpp>

#include "autograd/auto_context.hpp"

namespace ttml::utils {
using namespace ttnn::graph;

namespace MemoryUsageTracker {

constexpr double KB = 1024.0;
constexpr double MB = KB * KB;

// Global variables for memory usage tracking
static bool is_capture_active = false;

static std::shared_ptr<ttnn::graph::GraphProcessor> graph_processor;
static tt::tt_metal::IGraphProcessor::RunMode capture_mode = tt::tt_metal::IGraphProcessor::RunMode::NORMAL;

static std::vector<std::string> trace_order;
static std::unordered_map<std::string, nlohmann::json> traces;

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
        if (traces.find(name) != traces.end()) {
            fmt::print("WARNING: Trace '{}' already exists, overwriting\n", name);
        }
        auto trace = graph_processor->end_graph_capture();
        traces[name] = trace;
        trace_order.push_back(name);
        is_capture_active = false;
    }
}

void snapshot(const std::string& name) {
    if (!is_capture_active) {
        throw std::runtime_error("MemoryUsageTracker: Cannot snapshot - capture is not active");
    }

    if (traces.find(name) != traces.end()) {
        fmt::print("WARNING: Snapshot '{}' already exists, overwriting\n", name);
    }

    // End current capture and save with the given name
    auto trace = graph_processor->end_graph_capture();
    traces[name] = trace;
    trace_order.push_back(name);

    // Start a new capture
    graph_processor = std::make_shared<ttnn::graph::GraphProcessor>(capture_mode);
    graph_processor->begin_graph_capture(capture_mode);
}

DRAMUsage get_dram_usage(const std::string& name) {
    auto it = traces.find(name);
    if (it == traces.end()) {
        throw std::runtime_error(fmt::format("MemoryUsageTracker: Trace '{}' not found", name));
    }
    return ttnn::graph::extract_dram_usage(it->second);
}

std::vector<std::pair<std::string, DRAMUsage>> get_dram_usage_all() {
    std::vector<std::pair<std::string, DRAMUsage>> result;
    for (const auto& name : trace_order) {
        result.push_back(std::make_pair(name, get_dram_usage(name)));
    }
    return result;
}

L1UsagePerCore get_l1_usage(const std::string& name) {
    auto it = traces.find(name);
    if (it == traces.end()) {
        throw std::runtime_error(fmt::format("MemoryUsageTracker: Trace '{}' not found", name));
    }
    return ttnn::graph::extract_resource_usage_per_core(it->second);
}

std::vector<std::pair<std::string, L1UsagePerCore>> get_l1_usage_all() {
    std::vector<std::pair<std::string, L1UsagePerCore>> result;
    for (const auto& name : trace_order) {
        result.push_back(std::make_pair(name, get_l1_usage(name)));
    }
    return result;
}

std::vector<std::string> get_trace_names() {
    return trace_order;
}

void print_memory_usage() {
    if (trace_order.empty()) {
        fmt::print("WARNING: No traces captured\n");
        return;
    }

    fmt::print("=== Memory Usage Summary ===\n\n");

    // Track cumulative values across checkpoints
    // cumulative_current = sum of all changes up to this point (actual memory in use)
    // cumulative_peak = max(prev_cumulative_peak, prev_cumulative_current + this_segment_peak)
    long long cumulative_current = 0;
    long long cumulative_peak = 0;

    for (const auto& name : trace_order) {
        auto dram_usage = get_dram_usage(name);
        auto l1_usage = get_l1_usage(name);

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
    trace_order.clear();
    is_capture_active = false;
    graph_processor.reset();
}

}  // namespace MemoryUsageTracker

namespace DramFootprintTracker {

namespace {
bool is_tracking_active = false;

// The allocator reports per-bank; per-device bytes = value * num_banks (the same convention as ttnn's
// get_memory_view / get_available_device_memory_in_bytes).
DramFootprint to_device(const tt::tt_metal::DramFootprint& per_bank) {
    const auto& allocator = autograd::ctx().get_device().allocator();
    const auto num_banks = static_cast<uint64_t>(allocator->get_num_banks(tt::tt_metal::BufferType::DRAM));
    return {
        .peak_allocated_bytes = per_bank.peak_allocated_bytes * num_banks,
        .min_largest_free_bytes = per_bank.min_largest_free_bytes * num_banks,
    };
}
}  // namespace

void begin() {
    if (is_tracking_active) {
        throw std::runtime_error("DramFootprintTracker: tracking is already active (begin called twice; not nestable)");
    }
    autograd::ctx().get_device().allocator()->begin_dram_footprint_tracking();
    is_tracking_active = true;
}

DramFootprint footprint() {
    if (!is_tracking_active) {
        return {};
    }
    return to_device(autograd::ctx().get_device().allocator()->get_dram_footprint());
}

DramFootprint end() {
    if (!is_tracking_active) {
        fmt::print("WARNING: DramFootprintTracker::end() called while tracking is not active\n");
        return {};
    }
    const auto result = to_device(autograd::ctx().get_device().allocator()->end_dram_footprint_tracking());
    is_tracking_active = false;
    return result;
}

}  // namespace DramFootprintTracker

DramFootprintScope::DramFootprintScope() {
    DramFootprintTracker::begin();
}
DramFootprintScope::~DramFootprintScope() {
    DramFootprintTracker::end();
}
DramFootprint DramFootprintScope::footprint() const {
    return DramFootprintTracker::footprint();
}

uint64_t dram_reserved_bytes() {
    const auto& allocator = autograd::ctx().get_device().allocator();
    const auto num_banks = static_cast<uint64_t>(allocator->get_num_banks(tt::tt_metal::BufferType::DRAM));
    return static_cast<uint64_t>(allocator->get_dram_reserved_bytes()) * num_banks;
}

uint64_t dram_arena_bytes() {
    const auto& allocator = autograd::ctx().get_device().allocator();
    const auto num_banks = static_cast<uint64_t>(allocator->get_num_banks(tt::tt_metal::BufferType::DRAM));
    return static_cast<uint64_t>(allocator->get_bank_size(tt::tt_metal::BufferType::DRAM)) * num_banks;
}

}  // namespace ttml::utils
