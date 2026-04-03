// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Standalone test for the matmul_block compute helper (matmul_block_helpers.hpp).
// Exercises the helper directly using the blocked reader/writer kernels.
// Tests single-block and multi-block (spill/reload) paths.

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
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include "mesh_dispatch_fixture.hpp"

namespace tt::tt_metal {

using namespace tt::constants;

namespace test_matmul_block_helper {

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
                acc += static_cast<float>(a[(i * K) + k]) * static_cast<float>(b[(k * N) + j]);
            }
            output[(i * N) + j] = bfloat16(acc);
        }
    }
}

// Rearrange tilized A matrix from row-major tile order to block-column order.
// The blocked reader reads contiguous blocks along the K dimension.
// For single-block (num_blocks=1), this is a no-op (same order).
static std::vector<bfloat16> reorder_tiles_to_block_column(
    const std::vector<bfloat16>& tilized, uint32_t Mt, uint32_t Kt, uint32_t block_w) {
    const uint32_t tiles_per_tile = TILE_HEIGHT * TILE_WIDTH;
    uint32_t num_blocks = Kt / block_w;
    std::vector<bfloat16> result(tilized.size());

    uint32_t dst_offset = 0;
    for (uint32_t blk = 0; blk < num_blocks; blk++) {
        for (uint32_t row = 0; row < Mt; row++) {
            for (uint32_t col = 0; col < block_w; col++) {
                uint32_t src_tile_idx = (row * Kt) + (blk * block_w) + col;
                uint32_t src_offset = src_tile_idx * tiles_per_tile;
                std::copy(
                    tilized.begin() + src_offset,
                    tilized.begin() + src_offset + tiles_per_tile,
                    result.begin() + dst_offset);
                dst_offset += tiles_per_tile;
            }
        }
    }
    return result;
}

struct BlockMatmulConfig {
    uint32_t M;               // rows (elements, must be multiple of 32)
    uint32_t N;               // cols (elements, must be multiple of 32)
    uint32_t K;               // inner dim (elements, must be multiple of 32)
    uint32_t out_subblock_h;  // output sub-block height in tiles
    uint32_t out_subblock_w;  // output sub-block width in tiles
    uint32_t in0_block_w;     // K-dimension block size in tiles
};

static bool run_matmul_block_helper_test(
    MeshDispatchFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const BlockMatmulConfig& cfg,
    float pcc_threshold = 0.97f) {
    uint32_t Mt = cfg.M / TILE_HEIGHT;
    uint32_t Kt = cfg.K / TILE_WIDTH;
    uint32_t Nt = cfg.N / TILE_WIDTH;
    uint32_t in0_block_w = cfg.in0_block_w;
    uint32_t out_subblock_h = cfg.out_subblock_h;
    uint32_t out_subblock_w = cfg.out_subblock_w;

    uint32_t num_blocks = Kt / in0_block_w;
    uint32_t in0_num_subblocks = Mt / out_subblock_h;
    uint32_t in1_num_subblocks = Nt / out_subblock_w;
    uint32_t in0_block_num_tiles = Mt * in0_block_w;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
    uint32_t in1_block_num_tiles = Nt * in0_block_w;
    uint32_t in1_per_core_w = Nt;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;
    uint32_t num_output_tiles = Mt * Nt;

    uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    CoreCoord core({0, 0});

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program{};

    // DRAM buffers — single page containing entire matrix
    uint32_t dram_size_a = single_tile_size * Mt * Kt;
    uint32_t dram_size_b = single_tile_size * Kt * Nt;
    uint32_t dram_size_c = single_tile_size * Mt * Nt;

    distributed::DeviceLocalBufferConfig local_config_a{
        .page_size = dram_size_a, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::DeviceLocalBufferConfig local_config_b{
        .page_size = dram_size_b, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::DeviceLocalBufferConfig local_config_c{
        .page_size = dram_size_c, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};

    auto src0_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = dram_size_a}, local_config_a, mesh_device.get());
    auto src1_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = dram_size_b}, local_config_b, mesh_device.get());
    auto dst_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = dram_size_c}, local_config_c, mesh_device.get());

    // Circular buffers
    // in0: double-buffered blocks of A
    uint32_t in0_cb_tiles = in0_block_num_tiles * 2;
    CircularBufferConfig cb_in0_config =
        CircularBufferConfig(in0_cb_tiles * single_tile_size, {{CBIndex::c_0, cb_data_format}})
            .set_page_size(CBIndex::c_0, single_tile_size);
    CreateCircularBuffer(program, core, cb_in0_config);

    // in1: double-buffered blocks of B
    uint32_t in1_cb_tiles = in1_block_num_tiles * 2;
    CircularBufferConfig cb_in1_config =
        CircularBufferConfig(in1_cb_tiles * single_tile_size, {{CBIndex::c_1, cb_data_format}})
            .set_page_size(CBIndex::c_1, single_tile_size);
    CreateCircularBuffer(program, core, cb_in1_config);

    // out (c_16) and interm (c_24) share the same L1 address space.
    // Partials go to interm during K-blocking; final result goes to out.
    std::map<uint8_t, tt::DataFormat> partials_and_out_spec = {
        {CBIndex::c_24, cb_data_format}, {CBIndex::c_16, cb_data_format}};
    CircularBufferConfig cb_out_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, partials_and_out_spec)
            .set_page_size(CBIndex::c_24, single_tile_size)
            .set_page_size(CBIndex::c_16, single_tile_size);
    CreateCircularBuffer(program, core, cb_out_config);

    // Reader: blocked reader
    auto reader_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_blocked.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    // Writer: unswizzle writer (handles sub-block output ordering)
    auto writer_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unswizzle.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Compute: uses matmul_block helper via programming example kernel
    std::vector<uint32_t> compute_args = {
        in0_block_w,
        in0_num_subblocks,
        in0_block_num_tiles,
        in0_subblock_num_tiles,
        in1_num_subblocks,
        in1_block_num_tiles,
        in1_per_core_w,
        num_blocks,
        out_subblock_h,
        out_subblock_w,
        out_subblock_num_tiles,
        1  // batch
    };
    CreateKernel(
        program,
        "tt_metal/programming_examples/matmul/matmul_common/kernels/compute/bmm_large_block_zm.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_args});

    // Runtime args for reader
    uint32_t in0_block_size_bytes = in0_block_num_tiles * single_tile_size;
    uint32_t in1_block_size_bytes = in1_block_num_tiles * single_tile_size;
    SetRuntimeArgs(
        program,
        reader_id,
        core,
        {src0_dram->address(),
         0,  // bank_id
         src1_dram->address(),
         0,  // bank_id
         num_blocks,
         in0_block_num_tiles,
         in1_block_num_tiles,
         in0_block_size_bytes,
         in1_block_size_bytes});

    // Runtime args for writer (unswizzle)
    uint32_t stride_r = out_subblock_w * single_tile_size * (Nt / out_subblock_w);
    uint32_t stride_subblock_r = out_subblock_h * out_subblock_w * single_tile_size * (Nt / out_subblock_w);
    uint32_t stride_subblock_c = out_subblock_w * single_tile_size;
    SetRuntimeArgs(
        program,
        writer_id,
        core,
        {dst_dram->address(),
         0,  // bank_id
         out_subblock_h,
         out_subblock_w,
         in0_num_subblocks,  // num_sub_blocks_m
         in1_num_subblocks,  // num_sub_blocks_n
         stride_r,
         stride_subblock_r,
         stride_subblock_c});

    // Generate random input data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<bfloat16> src0_vec(cfg.M * cfg.K);
    std::vector<bfloat16> src1_vec(cfg.K * cfg.N);
    for (auto& v : src0_vec) {
        v = bfloat16(dist(rng));
    }
    for (auto& v : src1_vec) {
        v = bfloat16(dist(rng));
    }

    // CPU golden
    std::vector<bfloat16> golden_vec(cfg.M * cfg.N, bfloat16(0.0f));
    golden_matmul(src0_vec, src1_vec, golden_vec, cfg.M, cfg.N, cfg.K);

    // Tilize inputs
    auto src0_tilized = tilize_nfaces(src0_vec, cfg.M, cfg.K);
    auto src1_tilized = tilize_nfaces(src1_vec, cfg.K, cfg.N);

    // Rearrange A to block-column order for the blocked reader
    src0_tilized = reorder_tiles_to_block_column(src0_tilized, Mt, Kt, in0_block_w);

    // Upload data
    auto src0_packed = pack_bfloat16_vec_into_uint32_vec(src0_tilized);
    auto src1_packed = pack_bfloat16_vec_into_uint32_vec(src1_tilized);
    fixture->WriteBuffer(mesh_device, src0_dram, src0_packed);
    fixture->WriteBuffer(mesh_device, src1_dram, src1_packed);

    // Execute
    workload.add_program(device_range, std::move(program));
    fixture->RunProgram(mesh_device, workload);

    // Read back result
    std::vector<uint32_t> result_packed;
    fixture->ReadBuffer(mesh_device, dst_dram, result_packed);
    auto result_vec = unpack_uint32_vec_into_bfloat16_vec(result_packed);

    // Result is in tilized row-major tile order — untilize it
    result_vec = untilize_nfaces(result_vec, cfg.M, cfg.N);

    float pcc = pcc_bfloat16(golden_vec, result_vec);
    log_info(
        LogTest,
        "M={}, N={}, K={}, block_w={}, sub_h={}, sub_w={}, num_blocks={} — PCC = {:.6f} (threshold: {})",
        cfg.M,
        cfg.N,
        cfg.K,
        cfg.in0_block_w,
        cfg.out_subblock_h,
        cfg.out_subblock_w,
        num_blocks,
        pcc,
        pcc_threshold);

    return pcc > pcc_threshold;
}

}  // namespace test_matmul_block_helper

using test_matmul_block_helper::run_matmul_block_helper_test;

// Single block, 1×1 sub-blocks: simplest case (64×64×64)
TEST_F(MeshDispatchFixture, TensixMatmulBlockHelperSmallSingleBlock) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_block_helper_test(
            this, device, {.M = 64, .N = 64, .K = 64, .out_subblock_h = 2, .out_subblock_w = 2, .in0_block_w = 2}));
    }
}

// Single block, multiple sub-blocks in M dimension (128×64×64)
TEST_F(MeshDispatchFixture, TensixMatmulBlockHelperMultiSubblockM) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_block_helper_test(
            this, device, {.M = 128, .N = 64, .K = 64, .out_subblock_h = 2, .out_subblock_w = 2, .in0_block_w = 2}));
    }
}

// Multi-block (num_blocks=2): exercises spill/reload path (64×64×128)
TEST_F(MeshDispatchFixture, TensixMatmulBlockHelperMultiBlock) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_block_helper_test(
            this, device, {.M = 64, .N = 64, .K = 128, .out_subblock_h = 2, .out_subblock_w = 2, .in0_block_w = 2}));
    }
}

// Multiple sub-blocks in both M and N dimensions (128×128×64)
TEST_F(MeshDispatchFixture, TensixMatmulBlockHelperMultiSubblockBoth) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_block_helper_test(
            this, device, {.M = 128, .N = 128, .K = 64, .out_subblock_h = 2, .out_subblock_w = 2, .in0_block_w = 2}));
    }
}

// Single tile output sub-block (32×32×64, sub_h=1, sub_w=1)
TEST_F(MeshDispatchFixture, TensixMatmulBlockHelperSingleTileSubblock) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_block_helper_test(
            this, device, {.M = 32, .N = 32, .K = 64, .out_subblock_h = 1, .out_subblock_w = 1, .in0_block_w = 2}));
    }
}

}  // namespace tt::tt_metal
