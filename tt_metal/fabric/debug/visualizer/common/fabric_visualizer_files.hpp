// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// What the fabric visualizer's stages agree on about each other's files: each file's kind and version, its name, and
// the folder it lives in. The stage that writes a file and the stage that reads it both use these, so they can't
// drift. Header-only, with no dependencies, so every stage can include it whatever it links.
//
// Each fabric run's files sit in one folder, named by the run id rank 0 made at fabric init:
//   <logs_dir>/generated/fabric/runs/<run_id>/
//       fabric_manifest_rank_<r>_of_<n>.json    one per rank on this host

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

namespace tt::tt_fabric::visualizer {

// A manifest's file_kind and manifest_version. The version changes when readers can't take the new shape as it is.
inline constexpr std::string_view k_manifest_file_kind = "fabric_manifest";
inline constexpr uint32_t k_manifest_version = 1;

// One rank's manifest file name, by its rank in the whole job:
// fabric_manifest_rank_<mpi_rank + 1>_of_<world_size>.json.
inline std::string make_manifest_file_name(uint32_t mpi_rank, uint32_t world_size) {
    return "fabric_manifest_rank_" + std::to_string(mpi_rank + 1) + "_of_" + std::to_string(world_size) + ".json";
}

// The folder every fabric run's folder sits in.
inline std::filesystem::path make_runs_directory(const std::filesystem::path& logs_dir) {
    return logs_dir / "generated" / "fabric" / "runs";
}

// One fabric run's folder.
inline std::filesystem::path make_run_directory(const std::filesystem::path& logs_dir, std::string_view run_id) {
    return make_runs_directory(logs_dir) / run_id;
}

// The newest run's folder under `logs_dir` that holds `file_name`, such as one rank's manifest. A run id starts with
// the UTC time it was made, so the newest run is the last by name. Null when no run's folder holds it.
inline std::optional<std::filesystem::path> find_latest_run_directory(
    const std::filesystem::path& logs_dir, std::string_view file_name) {
    std::optional<std::filesystem::path> latest;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(make_runs_directory(logs_dir), error)) {
        if (entry.is_directory() && std::filesystem::exists(entry.path() / file_name) &&
            (!latest.has_value() || entry.path().filename() > latest->filename())) {
            latest = entry.path();
        }
    }
    return latest;
}

}  // namespace tt::tt_fabric::visualizer
