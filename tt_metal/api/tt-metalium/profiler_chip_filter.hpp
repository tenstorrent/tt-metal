// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <cstdlib>
#include <set>
#include <sstream>
#include <string>

namespace tt::tt_metal {

inline bool profiler_chip_filter_parsed = false;
inline std::set<uint32_t> profiler_chip_filter_set;
inline bool profiler_chip_filter_enabled = false;

inline bool should_profile_chip(uint32_t device_id) {
    if (!profiler_chip_filter_parsed) {
        const char* val = std::getenv("TT_METAL_PROFILER_FILTER_CHIPS");
        if (val != nullptr && val[0] != '\0') {
            profiler_chip_filter_enabled = true;
            std::istringstream ss(val);
            std::string token;
            while (std::getline(ss, token, ',')) {
                if (!token.empty()) {
                    profiler_chip_filter_set.insert(static_cast<uint32_t>(std::stoul(token)));
                }
            }
        }
        profiler_chip_filter_parsed = true;
    }
    if (!profiler_chip_filter_enabled) {
        return true;
    }
    return profiler_chip_filter_set.count(device_id) > 0;
}

}  // namespace tt::tt_metal
