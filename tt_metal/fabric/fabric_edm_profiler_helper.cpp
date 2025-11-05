// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fabric_edm_profiler_helper.hpp"

namespace tt::tt_fabric {

CodeProfilingTimerType convert_to_code_profiling_timer_type(const std::string& timer_str) {
    TT_FATAL(!timer_str.empty(), "Empty code profiling timer string provided");

    TT_FATAL(
        CodeProfilingTimerTypeMap.contains(timer_str), "Invalid code profiling timer string provided: {}", timer_str);

    return CodeProfilingTimerTypeMap.at(timer_str);
}

std::string convert_code_profiling_timer_type_to_str(const CodeProfilingTimerType& timer_type) {
    auto it = std::find_if(
        CodeProfilingTimerTypeMap.begin(), CodeProfilingTimerTypeMap.end(), [&timer_type](const auto& entry) {
            return entry.second == timer_type;
        });
    TT_FATAL(
        it != CodeProfilingTimerTypeMap.end(), "Code Profiling Timer Type not found in map, cannot convert to string");
    return it->first;
}

}  // namespace tt::tt_fabric
