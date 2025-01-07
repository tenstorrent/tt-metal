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


inline std::string get_profiler_artifacts_dir()
{
    std::string artifactDir = string(PROFILER_RUNTIME_ROOT_DIR);
    const auto PROFILER_ARTIFACTS_DIR = std::getenv("TT_METAL_PROFILER_DIR");
    if (PROFILER_ARTIFACTS_DIR != nullptr)
    {
        artifactDir = string(PROFILER_ARTIFACTS_DIR) + "/";
    }
    return artifactDir;
}

inline std::string get_profiler_logs_dir()
{
    return get_profiler_artifacts_dir() + string(PROFILER_LOGS_DIR_NAME);
}

inline std::string PROFILER_ZONE_SRC_LOCATIONS_LOG =  get_profiler_logs_dir() + "zone_src_locations.log";
}  // namespace tt_metal

}  // namespace tt
