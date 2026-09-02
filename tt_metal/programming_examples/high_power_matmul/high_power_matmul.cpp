// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <fmt/core.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

struct RunReport {
    uint32_t grid_x = 0;
    uint32_t grid_y = 0;
    uint32_t active_cores = 0;
    std::string tiles_per_core;
    double elapsed_s = 0.0;
    double tflops = 0.0;
    double per_iter_ms = 0.0;
    std::string start_time;
    std::string end_time;
};

struct GridCandidate {
    uint32_t x = 0;
    uint32_t y = 0;
};

static std::string format_time(std::chrono::system_clock::time_point tp) {
    auto t = std::chrono::system_clock::to_time_t(tp);
    auto us =
        std::chrono::duration_cast<std::chrono::microseconds>(tp.time_since_epoch()) %
        std::chrono::seconds(1);

    std::tm tm_buf{};
#if defined(_WIN32)
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif

    char buffer[64];
    std::snprintf(
        buffer,
        sizeof(buffer),
        "%04d-%02d-%02d %02d:%02d:%02d.%06lld",
        tm_buf.tm_year + 1900,
        tm_buf.tm_mon + 1,
        tm_buf.tm_mday,
        tm_buf.tm_hour,
        tm_buf.tm_min,
        tm_buf.tm_sec,
        static_cast<long long>(us.count()));

    return std::string(buffer);
}

static std::vector<GridCandidate> build_test_grids(uint32_t max_x, uint32_t max_y) {
    const std::vector<GridCandidate> requested = {
        {3, 2},  // 6
        {3, 3},  // 9
        {3, 4},  // 12
        {4, 3},  // 12
        {4, 4},  // 16
        {4, 5},  // 20
        {5, 4},  // 25
        {5, 5},  // 25
        {5, 6},  // 30
        {6, 5},  // 36
        {6, 6},  // 36
        {6, 7},  // 42
        {7, 6},  // 42
        {7, 7},  // 49
    };

    std::vector<GridCandidate> result;
    result.reserve(requested.size() + 1);

    for (const auto& g : requested) {
        if (g.x <= max_x && g.y <= max_y) {
            result.push_back(g);
        }
    }

    const GridCandidate max_grid{max_x, max_y};
    const bool max_grid_already_present = std::any_of(
        result.begin(),
        result.end(),
        [&](const GridCandidate& g) {
            return g.x == max_grid.x && g.y == max_grid.y;
        });

    if (!max_grid_already_present) {
        result.push_back(max_grid);
    }

    return result;
}

int main(int argc, char* argv[]) {
    uint32_t M = 256;
    uint32_t N = 256;
    uint32_t K = 512;
    uint32_t num_iterations = 100000;
    uint32_t fixed_tiles_per_core = 0;  // 0 = split total work (default), >0 = each core does exactly this many tiles

    if (argc >= 2) { M = std::stoul(argv[1]); }
    if (argc >= 3) { N = std::stoul(argv[2]); }
    if (argc >= 4) { K = std::stoul(argv[3]); }
    if (argc >= 5) { num_iterations = std::stoul(argv[4]); }
    if (argc >= 6) { fixed_tiles_per_core = std::stoul(argv[5]); }

    TT_FATAL(M % TILE_HEIGHT == 0, "M ({}) must be divisible by TILE_HEIGHT ({})", M, TILE_HEIGHT);
    TT_FATAL(N % TILE_WIDTH == 0, "N ({}) must be divisible by TILE_WIDTH ({})", N, TILE_WIDTH);
    TT_FATAL(K % TILE_WIDTH == 0, "K ({}) must be divisible by TILE_WIDTH ({})", K, TILE_WIDTH);

    const uint32_t Mt = M / TILE_HEIGHT;
    const uint32_t Kt = K / TILE_WIDTH;
    const uint32_t Nt = N / TILE_WIDTH;
    const uint32_t total_output_tiles = Mt * Nt;

    const double flops_per_iter =
        2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
    const double total_flops = flops_per_iter * static_cast<double>(num_iterations);

    fmt::print("=== High Power Matmul Workload ===\n");
    fmt::print("Matrix: M={} N={} K={} (tiles: {}x{}x{})\n", M, N, K, Mt, Nt, Kt);
    fmt::print("Output tiles: {}  |  Iterations: {}\n", total_output_tiles, num_iterations);
    fmt::print("Math fidelity: HiFi4  |  Data format: Float16_b\n");
    if (fixed_tiles_per_core > 0) {
        fmt::print("Mode: FIXED per-core ({} tiles/core) — total work scales with core count\n", fixed_tiles_per_core);
    } else {
        fmt::print("Mode: SPLIT total work — tiles divided equally across cores\n");
    }
    fmt::print("Expected FLOPs per full run: {:.2e}\n\n", total_flops);

    try {
        constexpr int device_id = 0;
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        auto& cq = mesh_device->mesh_command_queue();

        auto max_core_grid = mesh_device->compute_with_storage_grid_size();
        const uint32_t max_x = max_core_grid.x;
        const uint32_t max_y = max_core_grid.y;

        fmt::print("Detected max compute grid: {}x{} ({} cores)\n", max_x, max_y, max_x * max_y);

        const auto test_grids = build_test_grids(max_x, max_y);

        fmt::print("Selected test grids:\n");
        for (const auto& g : test_grids) {
            fmt::print("  {}x{} ({})\n", g.x, g.y, g.x * g.y);
        }
        fmt::print("\n");

        fmt::print("Generating input data once and reusing it for all runs...\n");
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

        std::vector<bfloat16> src0_vec(M * K);
        std::vector<bfloat16> src1_vec(K * N);
        for (auto& v : src0_vec) {
            v = bfloat16(dist(rng));
        }
        for (auto& v : src1_vec) {
            v = bfloat16(dist(rng));
        }

        src0_vec = tilize_nfaces(src0_vec, M, K);
        src1_vec = tilize_nfaces(src1_vec, K, N);

        std::vector<RunReport> summary_reports;
        summary_reports.reserve(test_grids.size());

        for (size_t grid_idx = 0; grid_idx < test_grids.size(); ++grid_idx) {
            if (grid_idx > 0) {
                fmt::print("\nWaiting 5 seconds before next run...\n");
                std::this_thread::sleep_for(std::chrono::seconds(5));
            }

            CoreCoord core_grid{};
            core_grid.x = test_grids[grid_idx].x;
            core_grid.y = test_grids[grid_idx].y;

            fmt::print("\n============================================================\n");
            fmt::print(
                "Starting run for compute grid: {}x{} ({} cores)\n",
                core_grid.x,
                core_grid.y,
                core_grid.x * core_grid.y);
            fmt::print("============================================================\n");

            uint32_t num_cores, work_per_core1, work_per_core2;
            CoreRangeSet all_cores, core_group_1, core_group_2;

            if (fixed_tiles_per_core > 0) {
                TT_FATAL(
                    fixed_tiles_per_core <= total_output_tiles,
                    "fixed_tiles_per_core ({}) must not exceed total_output_tiles ({})",
                    fixed_tiles_per_core, total_output_tiles);
                num_cores     = core_grid.x * core_grid.y;
                all_cores     = CoreRangeSet({CoreRange({0, 0}, {core_grid.x - 1, core_grid.y - 1})});
                core_group_1  = all_cores;
                core_group_2  = CoreRangeSet();
                work_per_core1 = fixed_tiles_per_core;
                work_per_core2 = 0;
            } else {
                auto [nc, ac, cg1, cg2, wpc1, wpc2] = split_work_to_cores(core_grid, total_output_tiles);
                num_cores = nc; all_cores = ac;
                core_group_1 = cg1; core_group_2 = cg2;
                work_per_core1 = wpc1; work_per_core2 = wpc2;
            }

            Program program{};
            constexpr uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;

            distributed::DeviceLocalBufferConfig dram_config{
                .page_size = single_tile_size,
                .buffer_type = BufferType::DRAM};

            auto src0_dram = distributed::MeshBuffer::create(
                distributed::ReplicatedBufferConfig{.size = single_tile_size * Mt * Kt},
                dram_config,
                mesh_device.get());

            auto src1_dram = distributed::MeshBuffer::create(
                distributed::ReplicatedBufferConfig{.size = single_tile_size * Kt * Nt},
                dram_config,
                mesh_device.get());

            auto dst_dram = distributed::MeshBuffer::create(
                distributed::ReplicatedBufferConfig{.size = single_tile_size * Mt * Nt},
                dram_config,
                mesh_device.get());

            const auto cb_fmt = tt::DataFormat::Float16_b;
            constexpr uint32_t num_cb_tiles = 2;

            tt_metal::CreateCircularBuffer(
                program,
                all_cores,
                CircularBufferConfig(num_cb_tiles * single_tile_size, {{CBIndex::c_0, cb_fmt}})
                    .set_page_size(CBIndex::c_0, single_tile_size));

            tt_metal::CreateCircularBuffer(
                program,
                all_cores,
                CircularBufferConfig(num_cb_tiles * single_tile_size, {{CBIndex::c_1, cb_fmt}})
                    .set_page_size(CBIndex::c_1, single_tile_size));

            tt_metal::CreateCircularBuffer(
                program,
                all_cores,
                CircularBufferConfig(num_cb_tiles * single_tile_size, {{CBIndex::c_16, cb_fmt}})
                    .set_page_size(CBIndex::c_16, single_tile_size));

            std::vector<uint32_t> reader_ct_args;
            TensorAccessorArgs(*src0_dram).append_to(reader_ct_args);
            TensorAccessorArgs(*src1_dram).append_to(reader_ct_args);

            auto reader_id = tt_metal::CreateKernel(
                program,
                OVERRIDE_KERNEL_PREFIX "high_power_matmul/kernels/dataflow/reader_power.cpp",
                all_cores,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_1,
                    .noc = NOC::RISCV_1_default,
                    .compile_args = reader_ct_args});

            std::vector<uint32_t> writer_ct_args;
            TensorAccessorArgs(*dst_dram).append_to(writer_ct_args);

            auto writer_id = tt_metal::CreateKernel(
                program,
                OVERRIDE_KERNEL_PREFIX "high_power_matmul/kernels/dataflow/writer_power.cpp",
                all_cores,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args = writer_ct_args});

            auto compute_id = tt_metal::CreateKernel(
                program,
                OVERRIDE_KERNEL_PREFIX "high_power_matmul/kernels/compute/mm_power.cpp",
                all_cores,
                ComputeConfig{.math_fidelity = MathFidelity::HiFi4});

            uint32_t work_offset = 0;
            uint32_t core_linear_idx = 0;
            auto work_groups = {
                std::make_pair(core_group_1, work_per_core1),
                std::make_pair(core_group_2, work_per_core2)
            };

            for (const auto& [ranges, work_per_core] : work_groups) {
                for (const auto& range : ranges.ranges()) {
                    for (const auto& core : range) {
                        // In fixed mode each core starts at a different offset (wrapping) so
                        // cores hit different DRAM addresses and don't serialise on the same bank.
                        const uint32_t effective_offset = (fixed_tiles_per_core > 0)
                            ? (core_linear_idx * fixed_tiles_per_core) % total_output_tiles
                            : work_offset;

                        tt_metal::SetRuntimeArgs(
                            program, reader_id, core,
                            {src0_dram->address(), src1_dram->address(),
                             Mt, Kt, Nt,
                             effective_offset, work_per_core, num_iterations});

                        tt_metal::SetRuntimeArgs(
                            program, writer_id, core,
                            {dst_dram->address(), work_per_core, effective_offset, num_iterations});

                        tt_metal::SetRuntimeArgs(
                            program, compute_id, core,
                            {work_per_core, Kt, num_iterations});

                        if (fixed_tiles_per_core == 0) work_offset += work_per_core;
                        ++core_linear_idx;
                    }
                }
            }

            distributed::EnqueueWriteMeshBuffer(cq, src0_dram, src0_vec, false);
            distributed::EnqueueWriteMeshBuffer(cq, src1_dram, src1_vec, false);

            std::string tiles_per_core_str;
            if (fixed_tiles_per_core > 0) {
                tiles_per_core_str = std::to_string(fixed_tiles_per_core);
            } else {
                tiles_per_core_str = std::to_string(work_per_core1);
                if (work_per_core2 > 0) {
                    tiles_per_core_str += " / " + std::to_string(work_per_core2);
                }
            }

            fmt::print(
                "Active cores: {}  |  Tiles/core: {}\n",
                num_cores,
                tiles_per_core_str);

            fmt::print(
                "Running {} iterations of {}x{}x{} HiFi4 matmul on {} cores...\n",
                num_iterations,
                M,
                N,
                K,
                num_cores);

            distributed::MeshWorkload workload;
            distributed::MeshCoordinateRange device_range(mesh_device->shape());
            workload.add_program(device_range, std::move(program));

            auto sys_start = std::chrono::system_clock::now();
            auto t_start = std::chrono::high_resolution_clock::now();

            distributed::EnqueueMeshWorkload(cq, workload, false);

            std::vector<bfloat16> result_vec(Mt * Nt * TILE_HW);
            distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram, true);

            auto t_end = std::chrono::high_resolution_clock::now();
            auto sys_end = std::chrono::system_clock::now();

            const std::string start_time_str = format_time(sys_start);
            const std::string end_time_str = format_time(sys_end);

            const double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
            // In fixed mode, actual FLOPs scale with num_cores (each core does the same work).
            const double run_flops = (fixed_tiles_per_core > 0)
                ? 2.0 * fixed_tiles_per_core * num_cores * Kt * TILE_HEIGHT * TILE_WIDTH * num_iterations
                : total_flops;
            const double tflops = run_flops / elapsed_s / 1e12;
            const double per_iter_ms = elapsed_s * 1000.0 / static_cast<double>(num_iterations);

            fmt::print("\n=== Results for {}x{} ===\n", core_grid.x, core_grid.y);
            fmt::print("Start time:      {}\n", start_time_str);
            fmt::print("End time:        {}\n", end_time_str);
            fmt::print("Total time:      {:.3f} s\n", elapsed_s);
            fmt::print("Throughput:      {:.2f} TFLOPS\n", tflops);
            fmt::print("Per-iteration:   {:.3f} ms\n", per_iter_ms);
            fmt::print("Test Passed\n");

            summary_reports.push_back(RunReport{
                .grid_x = core_grid.x,
                .grid_y = core_grid.y,
                .active_cores = num_cores,
                .tiles_per_core = tiles_per_core_str,
                .elapsed_s = elapsed_s,
                .tflops = tflops,
                .per_iter_ms = per_iter_ms,
                .start_time = start_time_str,
                .end_time = end_time_str
            });
        }

        fmt::print("\n\n========================================================================================================================\n");
        fmt::print("SUMMARY REPORT\n");
        fmt::print("========================================================================================================================\n");
        fmt::print(
            "{:>8} {:>12} {:>16} {:>16} {:>16} {:>18} {:>26} {:>26}\n",
            "Grid",
            "Cores",
            "Tiles/Core",
            "Time [s]",
            "TFLOPS",
            "Per iter [ms]",
            "Start Time",
            "End Time");

        for (const auto& r : summary_reports) {
            fmt::print(
                "{:>3}x{:<3} {:>12} {:>16} {:>16.3f} {:>16.2f} {:>18.3f} {:>26} {:>26}\n",
                r.grid_x,
                r.grid_y,
                r.active_cores,
                r.tiles_per_core,
                r.elapsed_s,
                r.tflops,
                r.per_iter_ms,
                r.start_time,
                r.end_time);
        }

        mesh_device->close();

    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed: {}\n", e.what());
        throw;
    }

    return 0;
}