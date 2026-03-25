// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compile-stress benchmark for remote JIT compile server scaling.
//
// Creates N unique compute kernels (each with distinct compile-time args to
// avoid JIT cache reuse) and compiles them.  With a remote compile server the
// compilations fan out across hosts; the measured throughput shows the speedup.
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
#include <random>

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
    uint32_t& kernel_counter,
    uint32_t target_num_kernels,
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

    for (uint32_t y = 0; y < grid_y && kernel_counter < target_num_kernels; y++) {
        for (uint32_t x = 0; x < grid_x && kernel_counter < target_num_kernels; x++) {
            CreateKernel(program, kernel_path, CoreCoord(x, y), ComputeConfig{.compile_args = {kernel_counter, seed}});
            kernel_counter++;
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

    auto start = std::chrono::steady_clock::now();

    uint32_t total_kernels = 0;
    for (uint32_t p = 0; p < num_programs && total_kernels < target_num_kernels; p++) {
        auto program = create_compute_program(
            all_cores, single_tile_size, kernel_path, grid_x, grid_y, total_kernels, target_num_kernels, seed);

        auto prog_start = std::chrono::steady_clock::now();
        detail::CompileProgram(dev, program);
        auto prog_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - prog_start)
                .count();
        uint32_t kernels_this = std::min(cores_per_program, target_num_kernels - p * cores_per_program);
        log_info(LogTest, "  program {}/{}: {} kernels in {}ms", p + 1, num_programs, kernels_this, prog_ms);
    }

    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    double kernels_per_sec = elapsed_ms > 0 ? total_kernels * 1000.0 / elapsed_ms : 0;

    log_info(
        LogTest,
        "Compile stress result: {} unique compute kernels, {} programs, {}ms total ({:.1f} kernels/sec)",
        total_kernels,
        num_programs,
        elapsed_ms,
        kernels_per_sec);

    ASSERT_EQ(total_kernels, target_num_kernels);
}
