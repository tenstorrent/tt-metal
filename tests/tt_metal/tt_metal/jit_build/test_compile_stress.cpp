// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Compile-stress benchmark for the remote JIT compile server.
//
// Creates N compute kernels with distinct {id, seed} compile-time args so each
// one hashes uniquely and bypasses the JIT cache, spreads them across programs
// limited by the compute grid, and compiles all programs in parallel to keep
// the compile server saturated.
//
// Kernels are partitioned into a shared pool (compiled identically by every
// client via TT_METAL_COMPILE_STRESS_SHARED_SEED) and a private pool (unique
// to this client via TT_METAL_COMPILE_STRESS_SEED), letting the multi-client
// harness dial in workload overlap between clients.
//
// Mock-mode only: the fixture calls experimental::configure_mock_mode and the
// test asserts the cluster target is Mock, so multiple clients can run on one
// host without contending on UMD driver locks.
//
// Configuration (env vars):
//   TT_METAL_COMPILE_STRESS_NUM_KERNELS     total kernels                 (default 1000)
//   TT_METAL_COMPILE_STRESS_ARCH            wormhole_b0 | blackhole |
//                                             quasar                      (default wormhole_b0)
//   TT_METAL_COMPILE_STRESS_NUM_CHIPS       mock num_chips                (default 1)
//   TT_METAL_COMPILE_STRESS_SEED            per-client private seed       (default random)
//   TT_METAL_COMPILE_STRESS_SHARED_FRACTION float in [0, 1]; fraction of
//                                             kernels in the shared pool  (default 0.0)
//   TT_METAL_COMPILE_STRESS_SHARED_SEED     shared-pool seed; must match
//                                             across clients              (default 0)
//   TT_METAL_COMPILE_STRESS_CLIENT_ID       informational client tag      (default 0)
//   TT_METAL_COMPILE_STRESS_OUTPUT          if set, path for JSON result
//   TT_METAL_COMPILE_STRESS_T_ZERO_NS       harness-internal rendezvous;
//                                             unix-epoch ns to sleep until
//                                             after warmup                (default unset)
//
// Example (direct invocation, single client, multiple remote servers):
//   TT_METAL_JIT_SERVER_ENABLE=1 TT_METAL_JIT_SERVER_ENDPOINTS=hostA:9876,hostB:9876
//   TT_METAL_COMPILE_STRESS_NUM_KERNELS=20000 ./build/test/tt_metal/unit_tests_jit_build
//   --gtest_filter='*TensixCompileStress*'
// For multi-client multi-server runs, use tests/scripts/run_compile_stress_harness.py.

#include "common/mesh_dispatch_fixture.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <future>
#include <random>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <enchantum/enchantum.hpp>
#include <fmt/format.h>
#include <tt_stl/span.hpp>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/experimental/mock_device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "common/env_lib.hpp"
#include "common/tt_backend_api_types.hpp"
#include "impl/context/metal_context.hpp"

using namespace tt;
using namespace tt::tt_metal;

namespace {

// tt::parse_env<T> (common/env_lib.hpp) lacks a double specialization.
double parse_env_double(const char* name, double default_value) {
    const char* env = std::getenv(name);
    return env != nullptr ? std::stod(env) : default_value;
}

tt::ARCH get_mock_arch_from_env() {
    const char* env = std::getenv("TT_METAL_COMPILE_STRESS_ARCH");
    if (!env) {
        return tt::ARCH::WORMHOLE_B0;
    }
    tt::ARCH arch = tt::get_arch_from_string(env);
    TT_FATAL(arch != tt::ARCH::Invalid, "Invalid TT_METAL_COMPILE_STRESS_ARCH value: '{}'", env);
    return arch;
}

constexpr std::string_view target_device_type_to_string(tt::TargetDevice t) noexcept {
    const std::string_view name = enchantum::to_string(t);
    return name.empty() ? std::string_view{"Unknown"} : name;
}

uint64_t now_unix_ns() {
    struct timespec ts{};
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

// Sleep until the requested CLOCK_REALTIME timestamp, yielding in the final
// sub-millisecond to land close to t_zero_ns.
void sleep_until_unix_ns(uint64_t t_zero_ns) {
    while (true) {
        uint64_t now = now_unix_ns();
        if (now >= t_zero_ns) {
            return;
        }
        uint64_t delta_ns = t_zero_ns - now;
        if (delta_ns > 1'000'000ULL) {
            std::this_thread::sleep_for(std::chrono::nanoseconds(delta_ns - 500'000ULL));
        } else {
            std::this_thread::yield();
        }
    }
}

Program create_compute_program(
    CoreRange core_range,
    uint32_t single_tile_size,
    const std::string& kernel_path,
    uint32_t grid_x,
    uint32_t grid_y,
    ttsl::Span<const std::array<uint32_t, 2>> compile_args_slice) {
    Program program = CreateProgram();

    CircularBufferConfig cb_src =
        CircularBufferConfig(8 * single_tile_size, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_0, single_tile_size);
    CreateCircularBuffer(program, core_range, cb_src);

    CircularBufferConfig cb_out =
        CircularBufferConfig(single_tile_size, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_16, single_tile_size);
    CreateCircularBuffer(program, core_range, cb_out);

    size_t k = 0;
    for (uint32_t y = 0; y < grid_y && k < compile_args_slice.size(); y++) {
        for (uint32_t x = 0; x < grid_x && k < compile_args_slice.size(); x++) {
            const auto& args = compile_args_slice[k];
            CreateKernel(program, kernel_path, CoreCoord(x, y), ComputeConfig{.compile_args = {args[0], args[1]}});
            k++;
        }
    }

    return program;
}

struct StressResult {
    uint32_t client_id;
    uint32_t private_seed;
    double shared_fraction;
    uint32_t shared_seed;
    uint32_t num_shared;
    std::string arch_name;
    uint32_t num_chips;
    uint32_t num_kernels;
    uint32_t num_programs;
    std::string_view target_device_type;
    uint64_t start_unix_ns;
    uint64_t end_unix_ns;
    double total_elapsed_ms;
};

void write_result_json(const std::string& path, const StressResult& r) {
    std::ofstream out(path);
    TT_FATAL(out.is_open(), "Cannot open {} for writing stress result JSON", path);
    out << "{\n";
    out << "  \"client_id\": " << r.client_id << ",\n";
    out << "  \"private_seed\": " << r.private_seed << ",\n";
    out << "  \"shared_fraction\": " << fmt::format("{:.6f}", r.shared_fraction) << ",\n";
    out << "  \"shared_seed\": " << r.shared_seed << ",\n";
    out << "  \"num_shared\": " << r.num_shared << ",\n";
    out << R"(  "arch": ")" << r.arch_name << "\",\n";
    out << "  \"num_chips\": " << r.num_chips << ",\n";
    out << "  \"num_kernels\": " << r.num_kernels << ",\n";
    out << "  \"num_programs\": " << r.num_programs << ",\n";
    out << R"(  "target_device_type": ")" << r.target_device_type << "\",\n";
    out << "  \"start_unix_ns\": " << r.start_unix_ns << ",\n";
    out << "  \"end_unix_ns\": " << r.end_unix_ns << ",\n";
    out << "  \"total_elapsed_ms\": " << fmt::format("{:.3f}", r.total_elapsed_ms) << "\n";
    out << "}\n";
    out.close();
    TT_FATAL(!out.fail(), "Failed to write stress result JSON to {}", path);
}

}  // namespace

class CompileStressFixture : public MeshDispatchFixture {
protected:
    void SetUp() override {
        experimental::configure_mock_mode(
            get_mock_arch_from_env(), tt::parse_env<std::uint32_t>("TT_METAL_COMPILE_STRESS_NUM_CHIPS", 1));
        MeshDispatchFixture::SetUp();
    }
    void TearDown() override {
        MeshDispatchFixture::TearDown();
        experimental::disable_mock_mode();
    }
};

TEST_F(CompileStressFixture, DISABLED_TensixCompileStress) {
    const auto target = MetalContext::instance().get_cluster().get_target_device_type();
    TT_FATAL(
        target == tt::TargetDevice::Mock,
        "CompileStressFixture expects mock device; got target_type={}.",
        target_device_type_to_string(target));

    IDevice* dev = devices_[0]->get_devices()[0];

    const uint32_t target_num_kernels = tt::parse_env<std::uint32_t>("TT_METAL_COMPILE_STRESS_NUM_KERNELS", 1000);
    const uint32_t client_id = tt::parse_env<std::uint32_t>("TT_METAL_COMPILE_STRESS_CLIENT_ID", 0);
    const uint32_t num_chips = tt::parse_env<std::uint32_t>("TT_METAL_COMPILE_STRESS_NUM_CHIPS", 1);
    const std::string arch_name = tt::parse_env<std::string>("TT_METAL_COMPILE_STRESS_ARCH", "wormhole_b0");
    const std::string output_path = tt::parse_env<std::string>("TT_METAL_COMPILE_STRESS_OUTPUT", "");
    const uint64_t t_zero_ns = tt::parse_env<std::uint64_t>("TT_METAL_COMPILE_STRESS_T_ZERO_NS", 0);

    const uint32_t private_seed = [] {
        const char* env = std::getenv("TT_METAL_COMPILE_STRESS_SEED");
        if (env) {
            return static_cast<uint32_t>(std::stoul(env));
        }
        return static_cast<uint32_t>(std::random_device{}());
    }();

    const double shared_fraction =
        std::clamp(parse_env_double("TT_METAL_COMPILE_STRESS_SHARED_FRACTION", 0.0), 0.0, 1.0);
    const uint32_t shared_seed = tt::parse_env<std::uint32_t>("TT_METAL_COMPILE_STRESS_SHARED_SEED", 0);
    const uint32_t num_shared = static_cast<uint32_t>(std::floor(shared_fraction * target_num_kernels));
    const uint32_t num_private = target_num_kernels - num_shared;

    CoreCoord compute_grid = dev->compute_with_storage_grid_size();
    const uint32_t grid_x = compute_grid.x;
    const uint32_t grid_y = compute_grid.y;
    const uint32_t cores_per_program = grid_x * grid_y;
    const uint32_t num_programs = (target_num_kernels + cores_per_program - 1) / cores_per_program;

    log_info(
        LogTest,
        "Compile stress config: client_id={} target_kernels={} num_shared={} shared_fraction={:.3f} "
        "shared_seed={} private_seed={} grid={}x{} cores_per_program={} num_programs={}",
        client_id,
        target_num_kernels,
        num_shared,
        shared_fraction,
        shared_seed,
        private_seed,
        grid_x,
        grid_y,
        cores_per_program,
        num_programs);

    // Shared ids are [0, num_shared) with shared_seed; private ids are
    // [num_shared, target_num_kernels) with private_seed. Disjoint id ranges
    // keep (id, seed) tuples unique across clients.
    std::vector<std::array<uint32_t, 2>> compile_args_per_kernel(target_num_kernels);
    for (uint32_t i = 0; i < target_num_kernels; ++i) {
        compile_args_per_kernel[i] = {i, i < num_shared ? shared_seed : private_seed};
    }

    const uint32_t single_tile_size = 2 * 1024;
    const std::string kernel_path = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_3m.cpp";
    CoreRange all_cores(CoreCoord(0, 0), CoreCoord(grid_x - 1, grid_y - 1));

    // Warmup initialises JIT infrastructure and the remote-server connection.
    // UINT32_MAX as the id keeps this kernel's hash out of the production set.
    {
        Program warmup = CreateProgram();
        CircularBufferConfig cb0 =
            CircularBufferConfig(8 * single_tile_size, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
                .set_page_size(tt::CBIndex::c_0, single_tile_size);
        CreateCircularBuffer(warmup, CoreCoord(0, 0), cb0);
        CircularBufferConfig cb16 =
            CircularBufferConfig(single_tile_size, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}})
                .set_page_size(tt::CBIndex::c_16, single_tile_size);
        CreateCircularBuffer(warmup, CoreCoord(0, 0), cb16);
        CreateKernel(warmup, kernel_path, CoreCoord(0, 0), ComputeConfig{.compile_args = {UINT32_MAX, private_seed}});
        detail::CompileProgram(dev, warmup);
    }
    log_info(LogTest, "Warmup compile done");

    std::vector<Program> programs;
    programs.reserve(num_programs);
    for (uint32_t p = 0; p < num_programs; p++) {
        uint32_t id_begin = p * cores_per_program;
        uint32_t id_end = std::min(id_begin + cores_per_program, target_num_kernels);
        auto slice =
            ttsl::Span<const std::array<uint32_t, 2>>(compile_args_per_kernel.data() + id_begin, id_end - id_begin);
        programs.push_back(create_compute_program(all_cores, single_tile_size, kernel_path, grid_x, grid_y, slice));
    }

    // T_ZERO rendezvous: harness sets a wall-clock target so all clients enter
    // the timed section together. No-op when unset.
    if (t_zero_ns != 0) {
        uint64_t now_ns = now_unix_ns();
        if (t_zero_ns > now_ns) {
            log_info(LogTest, "Waiting {}ms until T_ZERO for rendezvous", (t_zero_ns - now_ns) / 1'000'000ULL);
        } else {
            log_warning(
                LogTest,
                "T_ZERO already passed by {}ms; starting timed section immediately",
                (now_ns - t_zero_ns) / 1'000'000ULL);
        }
        sleep_until_unix_ns(t_zero_ns);
    }

    const uint64_t start_unix_ns = now_unix_ns();
    auto start = std::chrono::steady_clock::now();

    std::vector<std::future<void>> futures;
    futures.reserve(num_programs);
    for (auto& program : programs) {
        futures.push_back(std::async(std::launch::async, [dev, &program] { detail::CompileProgram(dev, program); }));
    }
    for (auto& f : futures) {
        f.get();
    }

    const double total_elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    const uint64_t end_unix_ns = now_unix_ns();
    const double kernels_per_sec = total_elapsed_ms > 0.0 ? target_num_kernels * 1000.0 / total_elapsed_ms : 0.0;

    log_info(
        LogTest,
        "Compile stress result: client_id={} {} kernels ({} shared, {} private), {} programs, {:.1f}ms total "
        "({:.1f} kernels/sec)",
        client_id,
        target_num_kernels,
        num_shared,
        num_private,
        num_programs,
        total_elapsed_ms,
        kernels_per_sec);

    if (!output_path.empty()) {
        StressResult result{
            .client_id = client_id,
            .private_seed = private_seed,
            .shared_fraction = shared_fraction,
            .shared_seed = shared_seed,
            .num_shared = num_shared,
            .arch_name = arch_name,
            .num_chips = num_chips,
            .num_kernels = target_num_kernels,
            .num_programs = num_programs,
            .target_device_type = target_device_type_to_string(target),
            .start_unix_ns = start_unix_ns,
            .end_unix_ns = end_unix_ns,
            .total_elapsed_ms = total_elapsed_ms,
        };
        write_result_json(output_path, result);
        log_info(LogTest, "Wrote stress result JSON to {}", output_path);
    }

    ASSERT_EQ(static_cast<uint32_t>(programs.size()), num_programs);
}
