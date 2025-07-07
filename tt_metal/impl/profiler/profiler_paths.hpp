// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define HOST_SIDE_LOG "profile_log_host.csv"
#define DEVICE_SIDE_LOG "profile_log_device.csv"

namespace tt {

namespace tt_metal {

constexpr std::string_view PROFILER_RUNTIME_ROOT_DIR = "generated/profiler/";
constexpr std::string_view PROFILER_LOGS_DIR_NAME = ".logs/";

inline std::string get_profiler_artifacts_dir() {
    std::string artifacts_dir;
    if (std::getenv("TT_METAL_PROFILER_DIR")) {
        artifacts_dir = std::string(std::getenv("TT_METAL_PROFILER_DIR")) + "/";
    } else {
        std::string prefix;
        if (std::getenv("TT_METAL_HOME")) {
            prefix = std::string(std::getenv("TT_METAL_HOME")) + "/";
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
}  // namespace tt_metal

}  // namespace tt
