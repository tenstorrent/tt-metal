// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compile-stress benchmark for remote JIT compile server scaling.
//
// Creates N unique compute kernels (each with distinct compile-time args to
// avoid JIT cache reuse) spread across multiple programs (limited by the
// per-core placement constraint), then compiles all programs in parallel so
// the JIT thread pool / remote compile server stays fully saturated.
//
// Configuration (env vars):
//   TT_METAL_COMPILE_STRESS_NUM_KERNELS  total unique kernels  (default 1000)
//
// Example:
//   TT_METAL_COMPILE_STRESS_NUM_KERNELS=40000 \
//       ./build/test/tt_metal/unit_tests_legacy \
//       --gtest_filter='*TensixCompileStress*'

#include "common/mesh_dispatch_fixture.hpp"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <random>
#include <vector>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace {

uint32_t get_env_uint32(const char* name, uint32_t default_value) {
    const char* env = std::getenv(name);
    if (env) {
        return static_cast<uint32_t>(std::stoul(env));
    }
    return default_value;
}

Program create_compute_program(
    CoreRange core_range,
    uint32_t single_tile_size,
    const std::string& kernel_path,
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t kernel_id_begin,
    uint32_t kernel_id_end,
    uint32_t seed) {
    Program program = CreateProgram();

    CircularBufferConfig cb_src =
        CircularBufferConfig(8 * single_tile_size, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_0, single_tile_size);
    CreateCircularBuffer(program, core_range, cb_src);

    CircularBufferConfig cb_out =
        CircularBufferConfig(single_tile_size, {{tt::CBIndex::c_16, tt::DataFormat::Float16_b}})
            .set_page_size(tt::CBIndex::c_16, single_tile_size);
    CreateCircularBuffer(program, core_range, cb_out);

    uint32_t id = kernel_id_begin;
    for (uint32_t y = 0; y < grid_y && id < kernel_id_end; y++) {
        for (uint32_t x = 0; x < grid_x && id < kernel_id_end; x++) {
            CreateKernel(program, kernel_path, CoreCoord(x, y), ComputeConfig{.compile_args = {id, seed}});
            id++;
        }
    }

    return program;
}

}  // namespace

TEST_F(MeshDispatchFixture, TensixCompileStress) {
    IDevice* dev = devices_[0]->get_devices()[0];

    const uint32_t target_num_kernels = get_env_uint32("TT_METAL_COMPILE_STRESS_NUM_KERNELS", 1000);

    std::random_device rd;
    const uint32_t seed = rd();

    CoreCoord compute_grid = dev->compute_with_storage_grid_size();
    const uint32_t grid_x = compute_grid.x;
    const uint32_t grid_y = compute_grid.y;
    const uint32_t cores_per_program = grid_x * grid_y;
    const uint32_t num_programs = (target_num_kernels + cores_per_program - 1) / cores_per_program;

    log_info(
        LogTest,
        "Compile stress config: target_kernels={} grid={}x{} cores_per_program={} num_programs={} seed={}",
        target_num_kernels,
        grid_x,
        grid_y,
        cores_per_program,
        num_programs,
        seed);

    const uint32_t single_tile_size = 2 * 1024;
    const std::string kernel_path = "tests/tt_metal/tt_metal/test_kernels/compute/eltwise_copy_3m.cpp";
    CoreRange all_cores(CoreCoord(0, 0), CoreCoord(grid_x - 1, grid_y - 1));

    // Warmup: compile a single kernel to initialise JIT infrastructure and
    // remote server connection before the timed section.
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
        CreateKernel(warmup, kernel_path, CoreCoord(0, 0), ComputeConfig{.compile_args = {UINT32_MAX, seed}});
        detail::CompileProgram(dev, warmup);
    }
    log_info(LogTest, "Warmup compile done, starting timed section");

    // Build all programs up front (not timed -- this is just host-side bookkeeping).
    std::vector<Program> programs;
    programs.reserve(num_programs);
    for (uint32_t p = 0; p < num_programs; p++) {
        uint32_t id_begin = p * cores_per_program;
        uint32_t id_end = std::min(id_begin + cores_per_program, target_num_kernels);
        programs.push_back(
            create_compute_program(all_cores, single_tile_size, kernel_path, grid_x, grid_y, id_begin, id_end, seed));
    }

    // Compile all programs in parallel so the JIT thread pool stays saturated.
    auto start = std::chrono::steady_clock::now();

    std::vector<std::future<void>> futures;
    futures.reserve(num_programs);
    for (auto& program : programs) {
        futures.push_back(std::async(std::launch::async, [dev, &program] { detail::CompileProgram(dev, program); }));
    }
    for (auto& f : futures) {
        f.get();
    }

    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    double kernels_per_sec = elapsed_ms > 0 ? target_num_kernels * 1000.0 / elapsed_ms : 0;

    log_info(
        LogTest,
        "Compile stress result: {} unique compute kernels, {} programs (compiled in parallel), {}ms total ({:.1f} "
        "kernels/sec)",
        target_num_kernels,
        num_programs,
        elapsed_ms,
        kernels_per_sec);

    ASSERT_EQ(static_cast<uint32_t>(programs.size()), num_programs);
}
