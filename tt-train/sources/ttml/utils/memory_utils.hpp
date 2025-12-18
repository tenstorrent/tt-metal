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
    // device id -> peak memory usage in bytes between begin_capture and end_capture
    std::unordered_map<std::string, long long> peak;
    // device id -> current memory usage in bytes at the time of end_capture
    std::unordered_map<std::string, long long> current;
};

struct L1Usage {
    // device id -> peak circular buffer usage in bytes between begin_capture and end_capture
    std::unordered_map<std::string, long long> peak_cb;
    // device id -> peak L1 buffer usage in bytes between begin_capture and end_capture
    std::unordered_map<std::string, long long> peak_buffer;
    // device id -> peak total (cb + buffer) usage in bytes between begin_capture and end_capture
    std::unordered_map<std::string, long long> peak_total;
    // device id -> current L1 buffer usage in bytes at the time of end_capture
    std::unordered_map<std::string, long long> current;
};

DRAMUsage extract_DRAM_usage(const nlohmann::json& trace);
L1Usage extract_L1_usage(const nlohmann::json& trace);

namespace MemoryUsageTracker {
ttnn::ScopeGuard begin_capture();
void end_capture();
DRAMUsage get_DRAM_usage();
L1Usage get_L1_usage();
void print_memory_usage();
}  // namespace MemoryUsageTracker
}  // namespace ttml::utils
