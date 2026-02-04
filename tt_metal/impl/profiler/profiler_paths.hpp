// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <string>
#include <string_view>

#define HOST_SIDE_LOG "profile_log_host.csv"
#define DEVICE_SIDE_LOG "profile_log_device.csv"

namespace tt::tt_metal {

constexpr std::string_view PROFILER_RUNTIME_ROOT_DIR = "generated/profiler/";
constexpr std::string_view PROFILER_LOGS_DIR_NAME = ".logs/";
constexpr std::string_view PROFILER_DEVICE_PERF_REPORT_NAME = "cpp_device_perf_report.csv";

// TODO: This function should not be reading environment variables directly, it should use rtoptions.
inline std::string get_profiler_artifacts_dir() {
    std::string artifacts_dir;
    const char* profiler_dir = std::getenv("TT_METAL_PROFILER_DIR");
    if (profiler_dir) {
        artifacts_dir = std::string(profiler_dir) + "/";
    } else {
        std::string prefix;
        const char* metal_home = std::getenv("TT_METAL_HOME");
        if (metal_home) {
            prefix = std::string(metal_home) + "/";
        }
        artifacts_dir = prefix + std::string(PROFILER_RUNTIME_ROOT_DIR);
    }
    return artifacts_dir;
}

inline std::string get_profiler_logs_dir() {
    return get_profiler_artifacts_dir() + std::string(PROFILER_LOGS_DIR_NAME);
}

inline std::string PROFILER_ZONE_SRC_LOCATIONS_LOG = get_profiler_logs_dir() + "zone_src_locations.log";
inline std::string NEW_PROFILER_ZONE_SRC_LOCATIONS_LOG = get_profiler_logs_dir() + "new_zone_src_locations.log";
}  // namespace tt::tt_metal
