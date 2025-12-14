// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

namespace ttml::utils {

struct DRAMUsage {
    std::unordered_map<std::string, long long> peak;     // per device
    std::unordered_map<std::string, long long> current;  // at end of trace
};

struct L1Usage {
    std::unordered_map<std::string, long long> peak_cb;      // peak circular buffer usage per device
    std::unordered_map<std::string, long long> peak_buffer;  // peak L1 buffer usage per device
    std::unordered_map<std::string, long long> peak_total;   // peak total (cb + buffer) per device
    std::unordered_map<std::string, long long> current;      // current L1 buffer usage at end of trace per device
};

DRAMUsage extract_DRAM_usage(const nlohmann::json& trace);
L1Usage extract_L1_usage(const nlohmann::json& trace);

namespace MemoryUsageTracker {
void begin_capture();
void end_capture();
DRAMUsage get_DRAM_usage();
L1Usage get_L1_usage();
void print_memory_usage();
}  // namespace MemoryUsageTracker
}  // namespace ttml::utils
