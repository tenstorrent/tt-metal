// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// fabric_capture: captures one rank's fabric routers, where the rank's manifest says they are. Run it after the run
// exits, or beside a run that's stuck, one process per rank, such as through tt-run with the run's rank bindings. Every
// rank captured together needs the same --capture-id.
//
// Exit status: 0 when everything the manifest names on this rank was read, 1 when the capture was written with
// problems its record describes, 2 when it failed and wrote nothing.

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include <cxxopts.hpp>
#include <tt_stl/assert.hpp>

#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_files.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_reads.hpp"
#include "tt_metal/fabric/debug/visualizer/capture/fabric_capture_umd.hpp"
#include "tt_metal/fabric/debug/visualizer/common/fabric_visualizer_build_identity.hpp"
#include "tt_metal/fabric/debug/visualizer/common/fabric_visualizer_files.hpp"

namespace {

namespace capture = tt::tt_fabric::capture;
namespace visualizer = tt::tt_fabric::visualizer;

constexpr int k_exit_complete = 0;
constexpr int k_exit_problems_recorded = 1;
constexpr int k_exit_failed = 2;

std::optional<std::string> get_env(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::nullopt : std::optional<std::string>(value);
}

// This process's rank in the run's job and the job's size: tt-run's TT_RUN_RANK, else MPI's, else a job of one.
std::pair<uint32_t, uint32_t> get_rank_and_world_size() {
    std::optional<std::string> rank = get_env("TT_RUN_RANK");
    if (!rank.has_value()) {
        rank = get_env("OMPI_COMM_WORLD_RANK");
    }
    const std::optional<std::string> world_size = get_env("OMPI_COMM_WORLD_SIZE");
    return {
        static_cast<uint32_t>(std::stoul(rank.value_or("0"))),
        static_cast<uint32_t>(std::stoul(world_size.value_or("1"))),
    };
}

// This rank's manifest in the newest run that has one, under the logs folder the run wrote to: TT_METAL_LOGS_PATH,
// else the working folder, as tt-metal chooses it.
std::filesystem::path find_latest_manifest() {
    const std::filesystem::path logs_dir = get_env("TT_METAL_LOGS_PATH").value_or(std::filesystem::current_path());
    const auto [mpi_rank, world_size] = get_rank_and_world_size();
    const std::string file_name = visualizer::make_manifest_file_name(mpi_rank, world_size);
    const std::optional<std::filesystem::path> run_directory =
        visualizer::find_latest_run_directory(logs_dir, file_name);
    TT_FATAL(
        run_directory.has_value(),
        "Fabric capture: no run under {} has {}; give the manifest with --manifest",
        visualizer::make_runs_directory(logs_dir).string(),
        file_name);
    return *run_directory / file_name;
}

// Whether an error's message is already in the log.
bool was_logged(const std::exception& error) {
    const std::string_view what = error.what();
    return what.starts_with("TT_FATAL @") || what.starts_with("TT_THROW @");
}

int run_capture(const cxxopts::ParseResult& options) {
    const std::filesystem::path manifest_path = options.contains("manifest")
                                                    ? std::filesystem::path(options["manifest"].as<std::string>())
                                                    : find_latest_manifest();
    const capture::CaptureRequest request{
        .manifest_path = manifest_path,
        .capture_id =
            options.contains("capture-id") ? options["capture-id"].as<std::string>() : capture::make_capture_id(),
        .liveness_options =
            {
                .num_rounds = options["liveness-rounds"].as<uint32_t>(),
                .interval = std::chrono::milliseconds(options["liveness-interval-ms"].as<uint32_t>()),
            },
        .capture_git_sha = std::string(visualizer::get_build_identity().git_sha),
    };
    std::cerr << "fabric_capture: capturing " << manifest_path.string() << " as capture " << request.capture_id << "\n";

    // The run may have named its host with TT_METAL_CLUSTER_ID; the reader names it the same way.
    capture::UmdDeviceReader reader(get_env("TT_METAL_CLUSTER_ID"));
    const capture::ClusterFacts facts = capture::collect_umd_cluster_facts(reader.get_cluster());
    capture::SteadyCaptureClock clock;
    const capture::CaptureResult result = capture::capture_rank(request, reader, facts, clock);

    std::cerr << "fabric_capture: wrote " << result.paths.record_path.string() << " and its .bin\n";
    if (result.outcome == capture::CaptureOutcome::PROBLEMS_RECORDED) {
        std::cerr << "fabric_capture: not everything the manifest names was read; the record says what's missing\n";
        return k_exit_problems_recorded;
    }
    return k_exit_complete;
}

}  // namespace

int main(int argc, char** argv) {
    cxxopts::Options parser("fabric_capture", "Captures one rank's fabric routers, where its manifest says they are.");
    parser.add_options()(
        "manifest",
        "This rank's manifest. Defaults to its manifest in the newest run under TT_METAL_LOGS_PATH (else the working "
        "folder), for its rank from TT_RUN_RANK or OMPI_COMM_WORLD_RANK",
        cxxopts::value<std::string>())(
        "capture-id",
        "The capture's id, which names its folder in the run's folder. Give every rank captured together the same one. "
        "Defaults to the UTC time now",
        cxxopts::value<std::string>())(
        "liveness-rounds",
        "How many times to sample every router before reading any core",
        cxxopts::value<uint32_t>()->default_value("3"))(
        "liveness-interval-ms", "The time between liveness rounds", cxxopts::value<uint32_t>()->default_value("1000"))(
        "h,help", "Print this help");

    try {
        const cxxopts::ParseResult options = parser.parse(argc, argv);
        if (options.contains("help")) {
            std::cout << parser.help() << "\n";
            return k_exit_complete;
        }
        return run_capture(options);
    } catch (const cxxopts::exceptions::exception& error) {
        std::cerr << "fabric_capture: " << error.what() << "; see --help\n";
        return k_exit_failed;
    } catch (const std::exception& error) {
        // Don't duplicate logged errors.
        if (was_logged(error)) {
            std::cerr << "fabric_capture: failed, and wrote no record\n";
        } else {
            std::cerr << "fabric_capture: failed, and wrote no record: " << error.what() << "\n";
        }
        return k_exit_failed;
    }
}
