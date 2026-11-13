// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// What the fabric visualizer's stages agree on about each other's files: each file's kind and version, its name, and
// the folder it lives in, and how a tool run beside the run finds its rank's files. The stage that writes a file and
// the stage that reads it both use these, so they can't drift. Header-only, with no dependencies, so every stage can
// include it whatever it links.
//
// Each fabric run's files sit in one folder, named by the run id rank 0 made at fabric init, and each capture of the
// run in a folder of its own, named by its capture id:
//   <logs_dir>/generated/fabric_debug/runs/<run_id>/
//       fabric_manifest_rank_<r>_of_<n>.json    one per rank on this host
//       captures/<capture_id>/
//           fabric_capture_rank_<r>_of_<n>.json  the capture record
//           fabric_capture_rank_<r>_of_<n>.bin   the bytes it read

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

namespace tt::tt_fabric::visualizer {

// A manifest's file_kind and manifest_version. The version changes when readers can't take the new shape as it is.
inline constexpr std::string_view k_manifest_file_kind = "fabric_manifest";
inline constexpr uint32_t k_manifest_version = 1;

// A capture record's file_kind and capture_version. The version changes as the manifest's does.
inline constexpr std::string_view k_capture_file_kind = "fabric_capture";
inline constexpr uint32_t k_capture_version = 1;

// One rank's manifest file name, by its rank in the whole job:
// fabric_manifest_rank_<mpi_rank + 1>_of_<world_size>.json.
inline std::string make_manifest_file_name(uint32_t mpi_rank, uint32_t world_size) {
    return "fabric_manifest_rank_" + std::to_string(mpi_rank + 1) + "_of_" + std::to_string(world_size) + ".json";
}

// The folder every fabric run's folder sits in.
inline std::filesystem::path make_runs_directory(const std::filesystem::path& logs_dir) {
    return logs_dir / "generated" / "fabric_debug" / "runs";
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

// A process's rank in its job, and the job's size.
struct JobRank {
    uint32_t mpi_rank = 0;
    uint32_t world_size = 1;
};

// This process's rank in its job: tt-run's TT_RUN_RANK, else MPI's OMPI_COMM_WORLD_RANK, else 0. The job's size is
// OMPI_COMM_WORLD_SIZE, else 1. A tool launched with the run's tt-run command gets the run's rank this way.
inline JobRank get_job_rank() {
    const char* rank = std::getenv("TT_RUN_RANK");
    if (rank == nullptr) {
        rank = std::getenv("OMPI_COMM_WORLD_RANK");
    }
    const char* world_size = std::getenv("OMPI_COMM_WORLD_SIZE");
    return JobRank{
        .mpi_rank = rank == nullptr ? 0 : static_cast<uint32_t>(std::stoul(rank)),
        .world_size = world_size == nullptr ? 1 : static_cast<uint32_t>(std::stoul(world_size)),
    };
}

// The logs folder tt-metal writes a run's files under: TT_METAL_LOGS_PATH, else the working folder.
inline std::filesystem::path get_logs_directory() {
    const char* logs_path = std::getenv("TT_METAL_LOGS_PATH");
    return logs_path == nullptr ? std::filesystem::current_path() : std::filesystem::path(logs_path);
}

// `rank`'s manifest in the newest run under `logs_dir` that holds one. Null when no run does.
inline std::optional<std::filesystem::path> find_latest_manifest(const std::filesystem::path& logs_dir, JobRank rank) {
    const std::string file_name = make_manifest_file_name(rank.mpi_rank, rank.world_size);
    const std::optional<std::filesystem::path> run_directory = find_latest_run_directory(logs_dir, file_name);
    if (!run_directory.has_value()) {
        return std::nullopt;
    }
    return *run_directory / file_name;
}

// One capture's folder in its run's folder. A run can be captured many times, and each capture's files sit in a
// folder named by its capture id.
inline std::filesystem::path make_capture_directory(
    const std::filesystem::path& run_directory, std::string_view capture_id) {
    return run_directory / "captures" / capture_id;
}

// One rank's capture file name without its extension, by its rank in the whole job.
inline std::string make_capture_file_stem(uint32_t mpi_rank, uint32_t world_size) {
    return "fabric_capture_rank_" + std::to_string(mpi_rank + 1) + "_of_" + std::to_string(world_size);
}

}  // namespace tt::tt_fabric::visualizer
