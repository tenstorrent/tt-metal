// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Standalone test for the matmul_block_fused_bias compute helper
// (matmul_block_fused_bias_helpers.hpp). Exercises the helper directly using
// hand-written reader/writer kernels. No TTNN op overhead.

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
namespace test_matmul_block_fused_bias {

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

// CPU golden: matmul + bias
static void golden_matmul_bias(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    const std::vector<bfloat16>& bias,
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
            acc += static_cast<float>(bias[j]);
            output[i * N + j] = bfloat16(acc);
        }
    }
}

// Run a single-core matmul+bias using the matmul_block_fused_bias compute helper.
//
// Parameters:
//   Mt, Nt, Kt — total output/inner dimensions in tiles
//   in0_block_w — K-dimension block width in tiles (= Kt / num_blocks)
//   out_subblock_h, out_subblock_w — sub-block dimensions in tiles
static bool run_matmul_block_fused_bias_test(
    MeshDispatchFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t Mt,
    uint32_t Nt,
    uint32_t Kt,
    uint32_t in0_block_w,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    float pcc_threshold = 0.97f) {
    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program{};
    CoreCoord core({0, 0});

    uint32_t M = Mt * TILE_HEIGHT;
    uint32_t N = Nt * TILE_WIDTH;
    uint32_t K = Kt * TILE_WIDTH;

    uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;

    // Derived block parameters
    uint32_t num_blocks = Kt / in0_block_w;  // number of K-dimension blocks
    uint32_t in0_num_subblocks = Mt / out_subblock_h;
    uint32_t in1_num_subblocks = Nt / out_subblock_w;
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
    uint32_t in1_block_num_tiles = out_subblock_w * in0_block_w * in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;
    uint32_t num_output_tiles = Mt * Nt;

    // DRAM buffers — use full-buffer page size (single page = single bank, no interleaving)
    // so the reader can use bank_id=0 and sequential addresses.
    uint32_t src0_size = single_tile_size * Mt * Kt;
    uint32_t src1_size = single_tile_size * Kt * Nt;
    uint32_t bias_size = single_tile_size * Nt;
    uint32_t dst_size = single_tile_size * Mt * Nt;

    auto src0_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = src0_size},
        distributed::DeviceLocalBufferConfig{
            .page_size = src0_size, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false},
        mesh_device.get());
    auto src1_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = src1_size},
        distributed::DeviceLocalBufferConfig{
            .page_size = src1_size, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false},
        mesh_device.get());
    auto bias_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = bias_size},
        distributed::DeviceLocalBufferConfig{
            .page_size = bias_size, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false},
        mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = dst_size},
        distributed::DeviceLocalBufferConfig{
            .page_size = dst_size, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false},
        mesh_device.get());

    // Circular buffers
    uint32_t cb_in0_id = 0;
    uint32_t cb_in1_id = 1;
    uint32_t cb_bias_id = 2;
    uint32_t cb_out_id = 16;
    uint32_t cb_interm_id = 24;

    // in0: double-buffered, need to hold at least in0_block_num_tiles
    uint32_t in0_cb_tiles = in0_block_num_tiles * 2;
    CircularBufferConfig cb_in0_config =
        CircularBufferConfig(in0_cb_tiles * single_tile_size, {{cb_in0_id, cb_data_format}})
            .set_page_size(cb_in0_id, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_in0_config);

    // in1: double-buffered
    uint32_t in1_cb_tiles = in1_block_num_tiles * 2;
    CircularBufferConfig cb_in1_config =
        CircularBufferConfig(in1_cb_tiles * single_tile_size, {{cb_in1_id, cb_data_format}})
            .set_page_size(cb_in1_id, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_in1_config);

    // bias: persistent, Nt tiles (one per output column tile)
    CircularBufferConfig cb_bias_config = CircularBufferConfig(Nt * single_tile_size, {{cb_bias_id, cb_data_format}})
                                              .set_page_size(cb_bias_id, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_bias_config);

    // interm and out share the same memory space
    std::map<uint8_t, tt::DataFormat> partials_and_out_spec = {
        {cb_interm_id, cb_data_format}, {cb_out_id, cb_data_format}};
    CircularBufferConfig cb_interm_out_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, partials_and_out_spec)
            .set_page_size(cb_interm_id, single_tile_size)
            .set_page_size(cb_out_id, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_interm_out_config);

    // Reader kernel — reads in0, in1, and bias from DRAM
    auto reader_id = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_blocked_with_bias.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    // Writer kernel — writes output sub-blocks to DRAM
    auto writer_id = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unswizzle.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Compute kernel — uses matmul_block_fused_bias helper
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
    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/matmul_block_fused_bias.cpp",
        core,
        tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_args});

    // Set runtime args
    // Reader: src0_addr, src0_bank, src1_addr, src1_bank, num_blocks,
    //         in0_block_tiles, in1_block_tiles, in0_block_bytes, in1_block_bytes,
    //         bias_addr, bias_bank, bias_tiles, bias_bytes
    tt_metal::SetRuntimeArgs(
        program,
        reader_id,
        core,
        {src0_dram_buffer->address(),
         0,
         src1_dram_buffer->address(),
         0,
         num_blocks,
         in0_block_num_tiles,
         in1_block_num_tiles,
         in0_block_num_tiles * single_tile_size,
         in1_block_num_tiles * single_tile_size,
         bias_dram_buffer->address(),
         0,
         Nt,
         Nt * single_tile_size});

    // Writer: dst_addr, dst_bank, inner_r, inner_c, num_sub_blocks_m, num_sub_blocks_n,
    //         stride_r, stride_subblock_r, stride_subblock_c
    // The helper outputs tiles in order: for each in0_subblock (M), for each in1_subblock (N),
    // sub-block tiles in row-major order. This matches writer_unswizzle's iteration order.
    tt_metal::SetRuntimeArgs(
        program,
        writer_id,
        core,
        {dst_dram_buffer->address(),
         0,
         out_subblock_h,                          // inner_r
         out_subblock_w,                          // inner_c
         in0_num_subblocks,                       // num_sub_blocks_m
         in1_num_subblocks,                       // num_sub_blocks_n
         Nt * single_tile_size,                   // stride_r: next row in full matrix
         out_subblock_h * Nt * single_tile_size,  // stride_subblock_r: next M-subblock row
         out_subblock_w * single_tile_size});     // stride_subblock_c: next N-subblock col

    // Generate random input data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<bfloat16> src0_vec(M * K);
    std::vector<bfloat16> src1_vec(K * N);
    // Bias is stored as Nt tiles; each tile is 32×32 but only first row is meaningful.
    // For add_bcast_rows, the entire tile gets broadcast, so we fill all 32 rows identically.
    std::vector<bfloat16> bias_vec_flat(N);  // 1×N bias values
    for (auto& v : src0_vec) {
        v = bfloat16(dist(rng));
    }
    for (auto& v : src1_vec) {
        v = bfloat16(dist(rng));
    }
    for (auto& v : bias_vec_flat) {
        v = bfloat16(dist(rng));
    }

    // Expand bias to tile format: create a 32×N matrix where each row is the bias vector.
    // The bias is a row vector [1, N] — tile [t] covers columns [t*32 .. t*32+31].
    // add_bcast_rows broadcasts each element across all rows within the tile.
    // Layout: 1 tile-row × Nt tile-columns = Nt tiles horizontally.
    std::vector<bfloat16> bias_tiled_vec(TILE_HEIGHT * N, bfloat16(0.0f));
    for (uint32_t r = 0; r < TILE_HEIGHT; r++) {
        for (uint32_t c = 0; c < N; c++) {
            bias_tiled_vec[r * N + c] = bias_vec_flat[c];
        }
    }

    // CPU golden reference: matmul + bias
    std::vector<bfloat16> golden_vec(M * N, bfloat16(0.0f));
    golden_matmul_bias(src0_vec, src1_vec, bias_vec_flat, golden_vec, M, N, K);

    // Tilize inputs for device
    auto src0_tilized = tilize_nfaces(src0_vec, M, K);
    auto src1_tilized = tilize_nfaces(src1_vec, K, N);
    auto bias_tilized = tilize_nfaces(bias_tiled_vec, TILE_HEIGHT, N);

    uint32_t tile_elems = TILE_HEIGHT * TILE_WIDTH;

    // Reorder A tiles into K-blocked layout:
    // The reader reads tiles sequentially per K-block. Within each block,
    // tiles are organized as: [subblock0 tiles][subblock1 tiles]...
    // Within a subblock: row-major (subblock_h rows × in0_block_w cols).
    //
    // Row-major tilize gives: tile(tr, tc) at index tr * Kt + tc
    // Blocked layout: block b contains tiles for K-cols [b*bw .. (b+1)*bw-1]
    //   Within block b: for each tile-row tr: tiles (tr, b*bw), (tr, b*bw+1), ...
    std::vector<bfloat16> src0_blocked(src0_tilized.size());
    {
        uint32_t dst_idx = 0;
        for (uint32_t b = 0; b < num_blocks; b++) {
            for (uint32_t tr = 0; tr < Mt; tr++) {
                for (uint32_t tc = b * in0_block_w; tc < (b + 1) * in0_block_w; tc++) {
                    uint32_t src_tile = tr * Kt + tc;
                    std::copy(
                        src0_tilized.begin() + src_tile * tile_elems,
                        src0_tilized.begin() + (src_tile + 1) * tile_elems,
                        src0_blocked.begin() + dst_idx * tile_elems);
                    dst_idx++;
                }
            }
        }
    }

    // Reorder B tiles into K-blocked layout:
    // Reader reads B tiles sequentially per K-block.
    // Row-major tilize gives: tile(tr, tc) at index tr * Nt + tc
    // Blocked layout: block b contains tiles for K-rows [b*bw .. (b+1)*bw-1]
    //   Within block b: for each K-tile-row kr: tiles (kr, 0), (kr, 1), ..., (kr, Nt-1)
    std::vector<bfloat16> src1_blocked(src1_tilized.size());
    {
        uint32_t dst_idx = 0;
        for (uint32_t b = 0; b < num_blocks; b++) {
            for (uint32_t tr = b * in0_block_w; tr < (b + 1) * in0_block_w; tr++) {
                for (uint32_t tc = 0; tc < Nt; tc++) {
                    uint32_t src_tile = tr * Nt + tc;
                    std::copy(
                        src1_tilized.begin() + src_tile * tile_elems,
                        src1_tilized.begin() + (src_tile + 1) * tile_elems,
                        src1_blocked.begin() + dst_idx * tile_elems);
                    dst_idx++;
                }
            }
        }
    }

    auto src0_packed = pack_bfloat16_vec_into_uint32_vec(src0_blocked);
    auto src1_packed = pack_bfloat16_vec_into_uint32_vec(src1_blocked);
    auto bias_packed = pack_bfloat16_vec_into_uint32_vec(bias_tilized);

    // Upload, execute, download
    fixture->WriteBuffer(mesh_device, src0_dram_buffer, src0_packed);
    fixture->WriteBuffer(mesh_device, src1_dram_buffer, src1_packed);
    fixture->WriteBuffer(mesh_device, bias_dram_buffer, bias_packed);
    workload.add_program(device_range, std::move(program));
    fixture->RunProgram(mesh_device, workload);

    std::vector<uint32_t> result_packed;
    fixture->ReadBuffer(mesh_device, dst_dram_buffer, result_packed);
    auto result_vec = unpack_uint32_vec_into_bfloat16_vec(result_packed);

    // Untilize result
    result_vec = untilize_nfaces(result_vec, M, N);

    // Check PCC
    float pcc = pcc_bfloat16(golden_vec, result_vec);
    log_info(
        LogTest,
        "Mt={}, Nt={}, Kt={}, in0_block_w={}, subblock={}x{} — PCC = {:.6f} (threshold: {})",
        Mt,
        Nt,
        Kt,
        in0_block_w,
        out_subblock_h,
        out_subblock_w,
        pcc,
        pcc_threshold);

    return pcc > pcc_threshold;
}

}  // namespace test_matmul_block_fused_bias

// Single K-block (no spill/reload), small: 2×2×2 tiles, subblock 2×2
TEST_F(MeshDispatchFixture, TensixMatmulBlockFusedBiasSmall) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(test_matmul_block_fused_bias::run_matmul_block_fused_bias_test(
            this,
            device,
            /*Mt=*/2,
            /*Nt=*/2,
            /*Kt=*/2,
            /*in0_block_w=*/2,
            /*out_subblock_h=*/2,
            /*out_subblock_w=*/2));
    }
}

// Multiple K-blocks (spill/reload): 4×2×4 tiles, 2 K-blocks of width 2
TEST_F(MeshDispatchFixture, TensixMatmulBlockFusedBiasMultiBlock) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(test_matmul_block_fused_bias::run_matmul_block_fused_bias_test(
            this,
            device,
            /*Mt=*/4,
            /*Nt=*/2,
            /*Kt=*/4,
            /*in0_block_w=*/2,
            /*out_subblock_h=*/2,
            /*out_subblock_w=*/2));
    }
}

// Rectangular output with multiple subblocks: 4×4×2 tiles, subblock 2×2
TEST_F(MeshDispatchFixture, TensixMatmulBlockFusedBiasRectangular) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(test_matmul_block_fused_bias::run_matmul_block_fused_bias_test(
            this,
            device,
            /*Mt=*/4,
            /*Nt=*/4,
            /*Kt=*/2,
            /*in0_block_w=*/2,
            /*out_subblock_h=*/2,
            /*out_subblock_w=*/2));
    }
}

// Single tile output: 1×1×2 tiles, 2 K-blocks
TEST_F(MeshDispatchFixture, TensixMatmulBlockFusedBiasSingleTile) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(test_matmul_block_fused_bias::run_matmul_block_fused_bias_test(
            this,
            device,
            /*Mt=*/1,
            /*Nt=*/1,
            /*Kt=*/2,
            /*in0_block_w=*/1,
            /*out_subblock_h=*/1,
            /*out_subblock_w=*/1));
    }
}

}  // namespace tt::tt_metal
