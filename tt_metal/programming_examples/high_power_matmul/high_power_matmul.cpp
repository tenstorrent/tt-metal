// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//
// High-power matmul workload for sustained power draw measurement.
//
// Runs a large HiFi4 matrix multiplication across all available cores for many
// iterations, producing near-maximum compute utilization and power consumption.
//
// Usage:
//   ./metal_example_high_power_matmul [M] [N] [K] [iterations]
//
// Defaults: M=4096 N=4096 K=2048 iterations=500
// All dimensions must be multiples of 32 (tile size).
//

#include <chrono>
#include <random>

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

int main(int argc, char* argv[]) {
    uint32_t M = 4096;
    uint32_t N = 4096;
    uint32_t K = 4096;
    uint32_t num_iterations = 500;

    if (argc >= 2) {
        M = std::stoul(argv[1]);
    }
    if (argc >= 3) {
        N = std::stoul(argv[2]);
    }
    if (argc >= 4) {
        K = std::stoul(argv[3]);
    }
    if (argc >= 5) {
        num_iterations = std::stoul(argv[4]);
    }

    TT_FATAL(M % TILE_HEIGHT == 0, "M ({}) must be divisible by TILE_HEIGHT ({})", M, TILE_HEIGHT);
    TT_FATAL(N % TILE_WIDTH == 0, "N ({}) must be divisible by TILE_WIDTH ({})", N, TILE_WIDTH);
    TT_FATAL(K % TILE_WIDTH == 0, "K ({}) must be divisible by TILE_WIDTH ({})", K, TILE_WIDTH);

    const uint32_t Mt = M / TILE_HEIGHT;
    const uint32_t Kt = K / TILE_WIDTH;
    const uint32_t Nt = N / TILE_WIDTH;
    const uint32_t total_output_tiles = Mt * Nt;

    double flops_per_iter = 2.0 * M * N * K;
    double total_flops = flops_per_iter * num_iterations;

    fmt::print("=== High Power Matmul Workload ===\n");
    fmt::print("Matrix: M={} N={} K={} (tiles: {}x{}x{})\n", M, N, K, Mt, Nt, Kt);
    fmt::print("Output tiles: {}  |  Iterations: {}\n", total_output_tiles, num_iterations);
    fmt::print("Math fidelity: HiFi4  |  Data format: Float16_b\n");
    fmt::print("Expected FLOPs: {:.2e}\n\n", total_flops);

    try {
        constexpr int device_id = 0;
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
        auto& cq = mesh_device->mesh_command_queue();

        auto core_grid = mesh_device->compute_with_storage_grid_size();
        fmt::print("Compute grid: {}x{} ({} cores)\n", core_grid.x, core_grid.y, core_grid.x * core_grid.y);

        auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2] =
            split_work_to_cores(core_grid, total_output_tiles);

        fmt::print("Active cores: {}  |  Tiles/core: {}", num_cores, work_per_core1);
        if (work_per_core2 > 0) {
            fmt::print(" / {}", work_per_core2);
        }
        fmt::print("\n\n");

        Program program{};
        constexpr uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;

        // DRAM buffers
        distributed::DeviceLocalBufferConfig dram_config{
            .page_size = single_tile_size, .buffer_type = BufferType::DRAM};

        auto src0_dram = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = single_tile_size * Mt * Kt}, dram_config, mesh_device.get());
        auto src1_dram = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = single_tile_size * Kt * Nt}, dram_config, mesh_device.get());
        auto dst_dram = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = single_tile_size * Mt * Nt}, dram_config, mesh_device.get());

        // Circular buffers (double-buffered)
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

        // Reader kernel
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

        // Writer kernel
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

        // Compute kernel (HiFi4)
        auto compute_id = tt_metal::CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "high_power_matmul/kernels/compute/mm_power.cpp",
            all_cores,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4});

        // Per-core runtime args
        uint32_t work_offset = 0;
        auto work_groups = {std::make_pair(core_group_1, work_per_core1), std::make_pair(core_group_2, work_per_core2)};

        for (const auto& [ranges, work_per_core] : work_groups) {
            for (const auto& range : ranges.ranges()) {
                for (const auto& core : range) {
                    tt_metal::SetRuntimeArgs(
                        program,
                        reader_id,
                        core,
                        {src0_dram->address(),
                         src1_dram->address(),
                         Mt,
                         Kt,
                         Nt,
                         work_offset,
                         work_per_core,
                         num_iterations});

                    tt_metal::SetRuntimeArgs(
                        program, writer_id, core, {dst_dram->address(), work_per_core, work_offset, num_iterations});

                    tt_metal::SetRuntimeArgs(program, compute_id, core, {work_per_core, Kt, num_iterations});

                    work_offset += work_per_core;
                }
            }
        }

        // Fill input buffers with random data
        fmt::print("Generating and uploading input data...\n");
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

        distributed::EnqueueWriteMeshBuffer(cq, src0_dram, src0_vec, false);
        distributed::EnqueueWriteMeshBuffer(cq, src1_dram, src1_vec, false);

        // Execute workload
        fmt::print(
            "Running {} iterations of {}x{}x{} HiFi4 matmul on {} cores...\n", num_iterations, M, N, K, num_cores);

        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range(mesh_device->shape());
        workload.add_program(device_range, std::move(program));

        auto t_start = std::chrono::high_resolution_clock::now();
        distributed::EnqueueMeshWorkload(cq, workload, false);

        // Blocking read to wait for completion
        std::vector<bfloat16> result_vec(Mt * Nt * TILE_HW);
        distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram, true);
        auto t_end = std::chrono::high_resolution_clock::now();

        double elapsed_s = std::chrono::duration<double>(t_end - t_start).count();
        double tflops = total_flops / elapsed_s / 1e12;

        fmt::print("\n=== Results ===\n");
        fmt::print("Total time:       {:.3f} s\n", elapsed_s);
        fmt::print("Throughput:       {:.2f} TFLOPS\n", tflops);
        fmt::print("Per-iteration:    {:.3f} ms\n", elapsed_s * 1000.0 / num_iterations);
        fmt::print("Test Passed\n");

        mesh_device->close();

    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed: {}\n", e.what());
        throw;
    }

    return 0;
}
