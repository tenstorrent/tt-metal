// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Isolated integration tests for TransposePreKBlock
// (ttnn/cpp/ttnn/kernel_lib/transpose_block_helpers.hpp), used as the
// PreKBlockFn slot of matmul_block.
//
// The functor WH-transposes in0 tiles from `in0_transpose_cb` into `in0_cb`
// once per K-block. The host-side golden mirrors by swapping row/col within
// each 32×32 tile of A before the plain-C++ matmul.
//
// Verify: matmul_block(A, B, TransposePreKBlock) == matmul_block(wh_xpose(A), B)
//   where wh_xpose transposes within each 32×32 tile of A.

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

namespace test_transpose_pre_k_block {

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

// Golden matmul with within-tile WH-transpose of A.
// A_eff[row, col] lives at (i*32 + r, j*32 + c) where (i,j) is the tile index
// and (r,c) is the within-tile position. WH-transpose swaps (r,c).
// So A_eff[i*32 + r, j*32 + c] = A_orig[i*32 + c, j*32 + r].
static void golden_matmul_wh_transpose_a(
    const std::vector<bfloat16>& a_orig,
    const std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    uint32_t M,
    uint32_t K,
    uint32_t N) {
    auto a_eff = [&](uint32_t row, uint32_t col) -> float {
        uint32_t tile_i = row / TILE_HEIGHT;
        uint32_t tile_j = col / TILE_WIDTH;
        uint32_t r = row % TILE_HEIGHT;
        uint32_t c = col % TILE_WIDTH;
        // Swap within-tile (r, c)
        uint32_t src_row = tile_i * TILE_HEIGHT + c;
        uint32_t src_col = tile_j * TILE_WIDTH + r;
        return static_cast<float>(a_orig[src_row * K + src_col]);
    };

    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; k++) {
                acc += a_eff(i, k) * static_cast<float>(b[k * N + j]);
            }
            output[i * N + j] = bfloat16(acc);
        }
    }
}

// Rearrange tilized A from row-major tile order to block-column order.
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

struct XposeConfig {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t out_subblock_h;
    uint32_t out_subblock_w;
    uint32_t in0_block_w;
};

static bool run_transpose_pre_k_block_test(
    MeshDispatchFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const XposeConfig& cfg,
    float pcc_threshold = 0.97f) {
    uint32_t Mt = cfg.M / TILE_HEIGHT;
    uint32_t Kt = cfg.K / TILE_WIDTH;
    uint32_t Nt = cfg.N / TILE_WIDTH;
    uint32_t in0_block_w = cfg.in0_block_w;

    uint32_t num_blocks = Kt / in0_block_w;
    uint32_t in0_num_subblocks = Mt / cfg.out_subblock_h;
    uint32_t in1_num_subblocks = Nt / cfg.out_subblock_w;
    uint32_t in0_block_num_tiles = Mt * in0_block_w;
    uint32_t in0_subblock_num_tiles = cfg.out_subblock_h * in0_block_w;
    uint32_t in1_block_num_tiles = Nt * in0_block_w;
    uint32_t in1_per_core_w = Nt;
    uint32_t out_subblock_num_tiles = cfg.out_subblock_h * cfg.out_subblock_w;
    uint32_t num_output_tiles = Mt * Nt;
    uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    CoreCoord core({0, 0});

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program{};

    uint32_t dram_size_a = single_tile_size * Mt * Kt;
    uint32_t dram_size_b = single_tile_size * Kt * Nt;
    uint32_t dram_size_c = single_tile_size * Mt * Nt;

    distributed::DeviceLocalBufferConfig lc_a{
        .page_size = dram_size_a, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::DeviceLocalBufferConfig lc_b{
        .page_size = dram_size_b, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::DeviceLocalBufferConfig lc_c{
        .page_size = dram_size_c, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};

    auto src0_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = dram_size_a}, lc_a, mesh_device.get());
    auto src1_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = dram_size_b}, lc_b, mesh_device.get());
    auto dst_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = dram_size_c}, lc_c, mesh_device.get());

    // CBs:
    //   c_3 (in0_transpose) is the reader target — original A.
    //   c_0 (in0) is what matmul_block consumes — populated by TransposePreKBlock.
    //   c_1 (in1) is B.
    //   c_16/c_24 share L1 per the standard multicast layout.
    uint32_t in0_cb_tiles = in0_block_num_tiles * 2;
    CircularBufferConfig cb_in0_cfg =
        CircularBufferConfig(in0_cb_tiles * single_tile_size, {{CBIndex::c_0, cb_data_format}})
            .set_page_size(CBIndex::c_0, single_tile_size);
    CreateCircularBuffer(program, core, cb_in0_cfg);

    CircularBufferConfig cb_in0_xp_cfg =
        CircularBufferConfig(in0_cb_tiles * single_tile_size, {{CBIndex::c_3, cb_data_format}})
            .set_page_size(CBIndex::c_3, single_tile_size);
    CreateCircularBuffer(program, core, cb_in0_xp_cfg);

    uint32_t in1_cb_tiles = in1_block_num_tiles * 2;
    CircularBufferConfig cb_in1_cfg =
        CircularBufferConfig(in1_cb_tiles * single_tile_size, {{CBIndex::c_1, cb_data_format}})
            .set_page_size(CBIndex::c_1, single_tile_size);
    CreateCircularBuffer(program, core, cb_in1_cfg);

    std::map<uint8_t, tt::DataFormat> partials_and_out_spec = {
        {CBIndex::c_24, cb_data_format}, {CBIndex::c_16, cb_data_format}};
    CircularBufferConfig cb_out_cfg = CircularBufferConfig(num_output_tiles * single_tile_size, partials_and_out_spec)
                                          .set_page_size(CBIndex::c_24, single_tile_size)
                                          .set_page_size(CBIndex::c_16, single_tile_size);
    CreateCircularBuffer(program, core, cb_out_cfg);

    // Custom reader that pushes A into c_3 (not c_0).
    auto reader_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_blocked_to_c3.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto writer_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unswizzle.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    std::vector<uint32_t> compute_args = {
        in0_block_w,
        in0_num_subblocks,
        in0_block_num_tiles,
        in0_subblock_num_tiles,
        in1_num_subblocks,
        in1_block_num_tiles,
        in1_per_core_w,
        num_blocks,
        cfg.out_subblock_h,
        cfg.out_subblock_w,
        out_subblock_num_tiles,
        1u};

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/test_transpose_pre_k_block_compute.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_args});

    uint32_t in0_block_size_bytes = in0_block_num_tiles * single_tile_size;
    uint32_t in1_block_size_bytes = in1_block_num_tiles * single_tile_size;
    SetRuntimeArgs(
        program,
        reader_id,
        core,
        {src0_dram->address(),
         0,
         src1_dram->address(),
         0,
         num_blocks,
         in0_block_num_tiles,
         in1_block_num_tiles,
         in0_block_size_bytes,
         in1_block_size_bytes});

    uint32_t stride_r = cfg.out_subblock_w * single_tile_size * (Nt / cfg.out_subblock_w);
    uint32_t stride_subblock_r = cfg.out_subblock_h * cfg.out_subblock_w * single_tile_size * (Nt / cfg.out_subblock_w);
    uint32_t stride_subblock_c = cfg.out_subblock_w * single_tile_size;
    SetRuntimeArgs(
        program,
        writer_id,
        core,
        {dst_dram->address(),
         0,
         cfg.out_subblock_h,
         cfg.out_subblock_w,
         in0_num_subblocks,
         in1_num_subblocks,
         stride_r,
         stride_subblock_r,
         stride_subblock_c});

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

    std::vector<bfloat16> golden(cfg.M * cfg.N, bfloat16(0.0f));
    golden_matmul_wh_transpose_a(src0_vec, src1_vec, golden, cfg.M, cfg.K, cfg.N);

    auto src0_tilized = tilize_nfaces(src0_vec, cfg.M, cfg.K);
    auto src1_tilized = tilize_nfaces(src1_vec, cfg.K, cfg.N);
    src0_tilized = reorder_tiles_to_block_column(src0_tilized, Mt, Kt, in0_block_w);

    auto src0_packed = pack_bfloat16_vec_into_uint32_vec(src0_tilized);
    auto src1_packed = pack_bfloat16_vec_into_uint32_vec(src1_tilized);
    fixture->WriteBuffer(mesh_device, src0_dram, src0_packed);
    fixture->WriteBuffer(mesh_device, src1_dram, src1_packed);

    workload.add_program(device_range, std::move(program));
    fixture->RunProgram(mesh_device, workload);

    std::vector<uint32_t> result_packed;
    fixture->ReadBuffer(mesh_device, dst_dram, result_packed);
    auto result_vec = unpack_uint32_vec_into_bfloat16_vec(result_packed);
    result_vec = untilize_nfaces(result_vec, cfg.M, cfg.N);

    float pcc = pcc_bfloat16(golden, result_vec);
    log_info(
        LogTest,
        "M={} N={} K={} blk_w={} sub_h={} sub_w={} nb={} — PCC = {:.6f} (thresh {:.4f})",
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

}  // namespace test_transpose_pre_k_block

using test_transpose_pre_k_block::run_transpose_pre_k_block_test;
using test_transpose_pre_k_block::XposeConfig;

TEST_F(MeshDispatchFixture, TensixTransposePreKBlockSmall) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_transpose_pre_k_block_test(
            this, device, {.M = 64, .N = 64, .K = 64, .out_subblock_h = 2, .out_subblock_w = 2, .in0_block_w = 2}));
    }
}

TEST_F(MeshDispatchFixture, TensixTransposePreKBlockMultiBlock) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_transpose_pre_k_block_test(
            this, device, {.M = 64, .N = 64, .K = 128, .out_subblock_h = 2, .out_subblock_w = 2, .in0_block_w = 2}));
    }
}

TEST_F(MeshDispatchFixture, TensixTransposePreKBlockMultiSubblockM) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_transpose_pre_k_block_test(
            this, device, {.M = 128, .N = 64, .K = 64, .out_subblock_h = 2, .out_subblock_w = 2, .in0_block_w = 2}));
    }
}

}  // namespace tt::tt_metal
