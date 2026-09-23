// SPDX-License-Identifier: Apache-2.0
// One process, two full opens. No device-opening test fixture precedes cold.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "jit_build/build_cache_telemetry.hpp"
#include "tools/profiler/tracy_debug_zones.hpp"

using json = nlohmann::json;
using tt::tt_metal::distributed::MeshDevice;
using tt::tt_metal::distributed::MeshDeviceConfig;
namespace fs = std::filesystem;

static size_t artifact_count(const fs::path& root) {
    size_t n = 0;
    if (fs::exists(root)) {
        for (const auto& e : fs::recursive_directory_iterator(root)) {
            if (e.is_regular_file() && (e.path().extension() == ".o" || e.path().extension() == ".elf")) ++n;
        }
    }
    return n;
}
static void save(const std::string& path, const json& result) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write metadata: " + path);
    f << result.dump(2) << '\n';
    if (!f) throw std::runtime_error("Metadata write failed");
}
int main(int argc, char** argv) {
    // Runner owns the arguments and supplies a fresh empty cache directory.
    std::string output, expected_arch;
    std::optional<size_t> expected_devices;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if ((arg == "--output" || arg == "--arch" || arg == "--devices") && i + 1 < argc) {
                const std::string value = argv[++i];
                if (arg == "--output") output = value;
                else if (arg == "--arch") expected_arch = value;
                else expected_devices = std::stoul(value);
            } else throw std::runtime_error("Unknown/incomplete argument: " + arg);
        }
        if (output.empty() || !expected_devices || (expected_arch != "wormhole_b0" && expected_arch != "blackhole"))
            throw std::runtime_error("Required: --output FILE --arch wormhole_b0|blackhole --devices N");
        const char* cache_env = std::getenv("TT_METAL_CACHE");
        if (!cache_env || !fs::is_directory(cache_env) || !fs::is_empty(cache_env))
            throw std::runtime_error("TT_METAL_CACHE must identify an existing empty job-owned directory");
        if (std::getenv("TT_METAL_CCACHE_KERNEL_SUPPORT"))
            throw std::runtime_error("Unset TT_METAL_CCACHE_KERNEL_SUPPORT; setting it to 0 still enables it");
#ifndef TRACY_ENABLE
        throw std::runtime_error("This benchmark requires a Tracy-enabled build");
#else
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        while (!TracyIsConnected) {
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("Tracy capture did not connect within 60 seconds");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
#endif
        // JIT telemetry is off for the measured opens. Cache validity is the artifact count.
        BuildCacheTelemetry::inst().disable();
        json result = {{"schema_version", 1}, {"pid", getpid()},
                       {"fabric_mode", "FABRIC_2D"}, {"build_type", FABRIC_INIT_BUILD_TYPE},
                       {"phases", json::object()}};
        save(output, result);
        // Set fabric once, before either open. Do not toggle the mode between phases.
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_2D);
        for (const std::string phase : {"cold", "hot"}) {
            const size_t artifacts_before = artifact_count(cache_env);
            if (phase == "hot" && artifacts_before == 0)
                throw std::runtime_error("Cold run left no .o/.elf cache artifacts for hot run");
            std::shared_ptr<MeshDevice> mesh;
            const auto start = std::chrono::steady_clock::now();
            if (phase == "cold") {
                TTZoneScopedDN(FABRIC_BUILDER, "FabricBuilderBenchmark::cold");
                mesh = MeshDevice::create(MeshDeviceConfig(std::nullopt));
            } else {
                TTZoneScopedDN(FABRIC_BUILDER, "FabricBuilderBenchmark::hot");
                mesh = MeshDevice::create(MeshDeviceConfig(std::nullopt));
            }
            const auto end = std::chrono::steady_clock::now();
            if (!mesh) throw std::runtime_error("Mesh creation returned null");
            const auto count = mesh->num_devices();
            const auto mesh_shape = mesh->shape();
            const auto dims = mesh_shape.dims();
            const std::vector<uint32_t> shape(dims.begin(), dims.end());
            auto device_ids = mesh->get_device_ids();
            std::sort(device_ids.begin(), device_ids.end());
            const auto arch = mesh->arch();
            const bool arch_ok = (expected_arch == "wormhole_b0" && arch == tt::ARCH::WORMHOLE_B0) ||
                                 (expected_arch == "blackhole" && arch == tt::ARCH::BLACKHOLE);
            json entry = {{"open_elapsed_ns", std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count()},
                          {"devices", count}, {"device_ids", device_ids}, {"shape", shape}, {"arch", expected_arch},
                          {"artifacts_before", artifacts_before}, {"artifacts_after", artifact_count(cache_env)}};
            // Close outside both phase marker zones; preserve caches, destroy handles.
            if (!mesh->close()) throw std::runtime_error("Mesh teardown reported failure");
            mesh.reset();
            entry["teardown_complete"] = true;
            result["phases"][phase] = entry;
            save(output, result);
            if (!arch_ok || count != *expected_devices)
                throw std::runtime_error("Unexpected architecture/device count; refusing partial/wrong-hardware benchmark");
        }
        result["completed"] = true;
        save(output, result);
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::DISABLED);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "fabric-builder benchmark failed: " << e.what() << '\n';
        return 2;
    }
}
