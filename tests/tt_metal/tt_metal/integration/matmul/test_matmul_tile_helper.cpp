// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Standalone test for the matmul_tile compute helper (matmul_tile_helpers.hpp).
// Exercises the helper directly using hand-written reader/writer kernels from the
// matmul_single_core programming example. No TTNN op overhead.

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include "mesh_dispatch_fixture.hpp"

namespace tt::tt_metal {

using namespace tt::constants;

// Pearson correlation coefficient for bfloat16 vectors
static float pcc_bfloat16(const std::vector<bfloat16>& a, const std::vector<bfloat16>& b) {
    float x_mean = 0.0f, y_mean = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        x_mean += static_cast<float>(a[i]);
        y_mean += static_cast<float>(b[i]);
    }
    x_mean /= a.size();
    y_mean /= b.size();

    float cov = 0.0f, x_var = 0.0f, y_var = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float xd = static_cast<float>(a[i]) - x_mean;
        float yd = static_cast<float>(b[i]) - y_mean;
        cov += xd * yd;
        x_var += xd * xd;
        y_var += yd * yd;
    }
    if (x_var == 0.0f || y_var == 0.0f) {
        return (x_var == y_var) ? 1.0f : 0.0f;
    }
    return cov / std::sqrt(x_var * y_var);
}

// CPU golden matmul for verification
static void golden_matmul(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K) {
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; k++) {
                acc += static_cast<float>(a[i * K + k]) * static_cast<float>(b[k * N + j]);
            }
            output[i * N + j] = bfloat16(acc);
        }
    }
}

// Run a single-core matmul using the matmul_tile compute helper and the original
// hand-written reader/writer from the programming example.
static bool run_matmul_tile_helper_test(
    MeshDispatchFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    float pcc_threshold = 0.97f) {
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program{};
    CoreCoord core({0, 0});

    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;

    uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;

    // DRAM buffers
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};

    auto src0_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = sizeof(bfloat16) * M * K}, dram_config, mesh_device.get());
    auto src1_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = sizeof(bfloat16) * K * N}, dram_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = sizeof(bfloat16) * M * N}, dram_config, mesh_device.get());

    // Circular buffers — 2 tiles for double buffering
    uint32_t num_tiles_per_cb = 2;
    uint32_t src0_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_tiles_per_cb * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_tiles_per_cb * single_tile_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t output_cb_index = CBIndex::c_16;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_tiles_per_cb * single_tile_size, {{output_cb_index, cb_data_format}})
            .set_page_size(output_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    // Reader kernel — original hand-written reader from programming example
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_dram_buffer).append_to(reader_compile_time_args);
    auto reader_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul/matmul_single_core/kernels/dataflow/reader_single_core_mm.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    // Writer kernel — original hand-written writer from programming example
    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
    auto writer_id = tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul/matmul_single_core/kernels/dataflow/writer_single_core_mm.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    // Compute kernel — uses matmul_tile helper
    std::vector<uint32_t> compute_compile_time_args = {Mt, Kt, Nt};
    tt_metal::CreateKernel(
        program,
        "tt_metal/programming_examples/matmul/matmul_single_core/kernels/compute/mm.cpp",
        core,
        tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_compile_time_args});

    // Set runtime args
    tt_metal::SetRuntimeArgs(
        program, reader_id, core, {src0_dram_buffer->address(), src1_dram_buffer->address(), Mt, Kt, Nt});
    tt_metal::SetRuntimeArgs(program, writer_id, core, {dst_dram_buffer->address(), Mt, Nt});

    // Generate random input data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<bfloat16> src0_vec(M * K);
    std::vector<bfloat16> src1_vec(K * N);
    for (auto& v : src0_vec) {
        v = bfloat16(dist(rng));
    }
    for (auto& v : src1_vec) {
        v = bfloat16(dist(rng));
    }

    // CPU golden reference
    std::vector<bfloat16> golden_vec(M * N, bfloat16(0.0f));
    golden_matmul(src0_vec, src1_vec, golden_vec, M, N, K);

    // Tilize inputs for device
    auto src0_tilized = tilize_nfaces(src0_vec, M, K);
    auto src1_tilized = tilize_nfaces(src1_vec, K, N);

    // Upload, execute, download
    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, src0_tilized, false);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, src1_tilized, false);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);

    std::vector<bfloat16> result_vec(M * N, bfloat16(0.0f));
    distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, true);

    // Untilize result
    result_vec = untilize_nfaces(result_vec, M, N);

    // Check PCC
    float pcc = pcc_bfloat16(golden_vec, result_vec);
    log_info(LogTest, "M={}, N={}, K={} — PCC = {:.6f} (threshold: {})", M, N, K, pcc, pcc_threshold);

    return pcc > pcc_threshold;
}

// Small tile-aligned shape: 64×64×64 (2×2×2 tiles)
TEST_F(MeshDispatchFixture, TensixMatmulTileHelperSmall) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_tile_helper_test(this, device, 64, 64, 64));
    }
}

// Rectangular shape: 128×64×96 (4×2×3 tiles)
TEST_F(MeshDispatchFixture, TensixMatmulTileHelperRectangular) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_tile_helper_test(this, device, 128, 64, 96));
    }
}

// Larger shape: 256×256×128 (8×8×4 tiles)
TEST_F(MeshDispatchFixture, TensixMatmulTileHelperLarger) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_tile_helper_test(this, device, 256, 256, 128));
    }
}

// Single tile: 32×32×32 (1×1×1 tiles) — minimum case
TEST_F(MeshDispatchFixture, TensixMatmulTileHelperSingleTile) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_tile_helper_test(this, device, 32, 32, 32));
    }
}

}  // namespace tt::tt_metal
