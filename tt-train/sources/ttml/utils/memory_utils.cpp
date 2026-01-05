// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "memory_utils.hpp"

#include "ttnn/graph/graph_processor.hpp"

namespace ttml::utils {
using namespace ttnn::graph;

namespace MemoryUsageTracker {
static std::shared_ptr<ttnn::graph::GraphProcessor> graph_processor;
static nlohmann::json trace;
static bool is_capture_active = false;

ttnn::ScopeGuard begin_capture() {
    if (is_capture_active) {
        throw std::runtime_error("MemoryUsageTracker: Capture already active");
    }
    auto mode = tt::tt_metal::IGraphProcessor::RunMode::NORMAL;
    graph_processor = std::make_shared<ttnn::graph::GraphProcessor>(mode);
    graph_processor->begin_graph_capture(mode);
    is_capture_active = true;
    return ttnn::make_guard([&]() { end_capture(); });
}

void end_capture() {
    if (is_capture_active) {
        trace = graph_processor->end_graph_capture();
        is_capture_active = false;
    }
}

DRAMUsage get_DRAM_usage() {
    if (trace.empty()) {
        fmt::print("WARNING: Calling get_DRAM_usage() before trace capture\n");
    }
    return ttnn::graph::extract_dram_usage(trace);
}

L1UsagePerCore get_L1_usage() {
    if (trace.empty()) {
        fmt::print("WARNING: Calling get_L1_usage() before trace capture\n");
    }
    return ttnn::graph::extract_resource_usage_per_core(trace, 1);
}

void print_memory_usage() {
    auto dram_usage = get_DRAM_usage();
    auto l1_usage = get_L1_usage();

    fmt::print("=== Memory Usage Summary ===\n");

    // Print DRAM usage
    fmt::print(
        "Peak DRAM {:.2f} MB, Current DRAM {:.2f} MB\n",
        dram_usage.peak / 1024.0 / 1024.0,
        dram_usage.current / 1024.0 / 1024.0);

    // Print L1 usage
    fmt::print(
        "Peak L1 CB {:.2f} MB, Peak L1 Buffer {:.2f} MB, Peak L1 Total {:.2f} MB\n",
        l1_usage.peak_cb / 1024.0 / 1024.0,
        l1_usage.peak_l1 / 1024.0 / 1024.0,
        l1_usage.peak_total / 1024.0 / 1024.0);
}
}  // namespace MemoryUsageTracker

}  // namespace ttml::utils
