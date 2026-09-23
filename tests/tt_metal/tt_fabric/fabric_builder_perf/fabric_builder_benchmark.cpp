// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Opens the full mesh twice in one process, first with an empty kernel cache (cold) and then with
// the cache the first open left behind (hot). Each open runs inside a Tracy zone so the fabric
// builder zones can be attributed to a phase. Results are written as JSON for
// test_fabric_builder_perf.py, which performs validation and golden comparison.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <enchantum/enchantum.hpp>
#include <nlohmann/json.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/cluster.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

#include "tests/tt_metal/test_utils/test_common.hpp"
#include "tools/profiler/tracy_debug_zones.hpp"

namespace {

namespace fs = std::filesystem;
using tt::tt_fabric::FabricConfig;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::MeshDeviceConfig;

constexpr auto TRACY_CONNECT_TIMEOUT = std::chrono::seconds(60);

struct BenchmarkArgs {
    fs::path output;
    FabricConfig fabric_config;
};

// Arg parsing.
BenchmarkArgs parse_args(int argc, char** argv) {
    std::vector<std::string> input_args(argv, argv + argc);
    TT_FATAL(
        test_args::has_command_option(input_args, "--output") &&
            test_args::has_command_option(input_args, "--fabric-config"),
        "Usage: {} --output FILE --fabric-config NAME",
        argv[0]);

    std::string output;
    std::string fabric_config_name;
    std::tie(output, input_args) = test_args::get_command_option_and_remaining_args(input_args, "--output");
    std::tie(fabric_config_name, input_args) =
        test_args::get_command_option_and_remaining_args(input_args, "--fabric-config");
    test_args::validate_remaining_args(input_args);

    const auto fabric_config = enchantum::cast<FabricConfig>(fabric_config_name);
    TT_FATAL(fabric_config.has_value(), "Unknown --fabric-config {}", fabric_config_name);
    return {output, fabric_config.value()};
}

// Kernel cache directory that we expect to be empty. 
// For cold cache profiling, an empty directory is expected. Hot cache
// profiling uses the same cache directory, except it will be populated with
// artifacts from the cold cache profiling.
fs::path get_kernel_cache_dir() {
    const char* cache_dir = std::getenv("TT_METAL_CACHE");
    TT_FATAL(cache_dir != nullptr, "TT_METAL_CACHE must be set to a fresh directory");
    TT_FATAL(
        fs::is_directory(cache_dir) && fs::is_empty(cache_dir),
        "TT_METAL_CACHE {} must be an empty directory",
        cache_dir);
    return cache_dir;
}

size_t count_cache_artifacts(const fs::path& cache_dir) {
    return std::count_if(
        fs::recursive_directory_iterator(cache_dir), fs::recursive_directory_iterator(), [](const auto& entry) {
            const auto extension = entry.path().extension();
            return entry.is_regular_file() && (extension == ".o" || extension == ".elf");
        });
}

// Returns after Tracy capture connects, or errors after timeout.
void wait_for_tracy_connection() {
    const auto deadline = std::chrono::steady_clock::now() + TRACY_CONNECT_TIMEOUT;
    while (!TracyIsConnected) {
        TT_FATAL(
            std::chrono::steady_clock::now() < deadline,
            "Tracy capture did not connect within {}s",
            TRACY_CONNECT_TIMEOUT.count());
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

// Cold cache profiling.
std::shared_ptr<MeshDevice> open_cold() {
    TTZoneScopedDN(FABRIC_BUILDER, "FabricBuilderBenchmark::cold");
    // null MeshDeviceConfig creates a mesh device with the shape of the connected system mesh
    return MeshDevice::create(MeshDeviceConfig(std::nullopt));
}

// Hot cache profiling.
std::shared_ptr<MeshDevice> open_hot() {
    TTZoneScopedDN(FABRIC_BUILDER, "FabricBuilderBenchmark::hot");
    // null MeshDeviceConfig creates a mesh device with the shape of the connected system mesh
    return MeshDevice::create(MeshDeviceConfig(std::nullopt));
}

struct PhaseResult {
    size_t num_devices;
    // Store artifacts before and after for cache condition validation
    size_t artifacts_before;
    size_t artifacts_after;
};

// Runs a phase of the benchmark. Opens and closes a mesh device.
PhaseResult run_phase(const fs::path& cache_dir, const std::function<std::shared_ptr<MeshDevice>()>& open_mesh) {
    const size_t artifacts_before = count_cache_artifacts(cache_dir);
    auto mesh = open_mesh();
    const size_t num_devices = mesh->num_devices();
    TT_FATAL(mesh->close(), "Mesh teardown failed");
    return {num_devices, artifacts_before, count_cache_artifacts(cache_dir)};
}

// Jsonification.
nlohmann::json to_json(const PhaseResult& phase) {
    return {{"artifacts_before", phase.artifacts_before}, {"artifacts_after", phase.artifacts_after}};
}

// Cluster discovery.
std::string get_cluster_type_name() {
    std::string name(enchantum::to_string(tt::tt_metal::GetClusterType()));
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);
    return name;
}

// Writes the collected results from the benchmark to a json file.
void write_results(const fs::path& output, const nlohmann::json& results) {
    std::ofstream file(output);
    TT_FATAL(file.is_open(), "Cannot open {} for writing", output.string());
    file << results.dump(2) << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    // Setup
    const auto args = parse_args(argc, argv);
    const auto cache_dir = get_kernel_cache_dir();
    wait_for_tracy_connection();
    tt::tt_fabric::SetFabricConfig(args.fabric_config);

    // Run the benchmark
    const PhaseResult cold = run_phase(cache_dir, open_cold);
    const PhaseResult hot = run_phase(cache_dir, open_hot);

    // Verify that the number of devices opened is the same for both phases
    TT_FATAL(
        cold.num_devices == hot.num_devices,
        "Cold opened {} devices but hot opened {}",
        cold.num_devices,
        hot.num_devices);

    // Write the results to a json file
    const nlohmann::json context = {
        {"arch", tt::tt_metal::hal::get_arch_name()},
        {"cluster_type", get_cluster_type_name()},
        {"num_devices", cold.num_devices},
        {"fabric_config", std::string(enchantum::to_string(args.fabric_config))},
    };
    const nlohmann::json phases = {{"cold", to_json(cold)}, {"hot", to_json(hot)}};
    write_results(args.output, {{"context", context}, {"phases", phases}});
    log_info(tt::LogTest, "Wrote fabric builder benchmark results to {}", args.output.string());

    // Teardown (each phase already closes the mesh device)
    tt::tt_fabric::SetFabricConfig(FabricConfig::DISABLED);
    return 0;
}
