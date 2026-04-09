// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Isolated integration tests for the matmul_block and add_bias_bcast_rows helpers.
// Exercises new helper features: PACKER_L1_ACC, PACK_RELU, FUSE_BIAS, and their
// combinations. Validates PCC against CPU golden reference.
//
// Phase 3, Instance 2 output — tests designed against the Phase 2 API design
// (docs/matmul_api_design.md). Requires Phase 3 Instance 1's helper implementation
// for the compute kernel to JIT-compile.

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

namespace test_matmul_helper_features {

// ─── PCC (Pearson Correlation Coefficient) ───────────────────────────────────

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

// ─── Golden reference: matmul + optional bias + optional relu ────────────────

static void golden_matmul(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    const std::vector<bfloat16>& bias,
    std::vector<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    bool has_bias,
    bool has_relu) {
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; k++) {
                acc += static_cast<float>(a[i * K + k]) * static_cast<float>(b[k * N + j]);
            }
            if (has_bias) {
                acc += static_cast<float>(bias[j]);
            }
            if (has_relu) {
                acc = std::max(0.0f, acc);
            }
            output[i * N + j] = bfloat16(acc);
        }
    }
}

// ─── Tile reordering for blocked reader ──────────────────────────────────────
// Rearranges tilized A matrix from row-major tile order to block-column order.
// The blocked reader reads contiguous blocks along the K dimension.

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

// ─── Test configuration ──────────────────────────────────────────────────────

struct FeatureTestConfig {
    uint32_t M;               // rows (elements, multiple of 32)
    uint32_t N;               // cols (elements, multiple of 32)
    uint32_t K;               // inner dim (elements, multiple of 32)
    uint32_t out_subblock_h;  // output sub-block height in tiles
    uint32_t out_subblock_w;  // output sub-block width in tiles
    uint32_t in0_block_w;     // K-dimension block size in tiles
    bool packer_l1_acc;       // enable PACKER_L1_ACC
    bool pack_relu;           // enable PACK_RELU
    bool fuse_bias;           // enable FUSE_BIAS (add_bias_bcast_rows)
};

// ─── Main test runner ────────────────────────────────────────────────────────

static bool run_matmul_helper_feature_test(
    MeshDispatchFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const FeatureTestConfig& cfg,
    float pcc_threshold = 0.97f) {
    // ── Compute tile dimensions ──
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

    // ── DRAM buffers ──
    uint32_t dram_size_a = single_tile_size * Mt * Kt;
    uint32_t dram_size_b = single_tile_size * Kt * Nt;
    uint32_t dram_size_c = single_tile_size * Mt * Nt;
    uint32_t dram_size_bias = single_tile_size * Nt;  // one tile-row of bias

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

    std::shared_ptr<distributed::MeshBuffer> bias_dram;
    if (cfg.fuse_bias) {
        distributed::DeviceLocalBufferConfig local_config_bias{
            .page_size = dram_size_bias, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
        bias_dram = distributed::MeshBuffer::create(
            distributed::ReplicatedBufferConfig{.size = dram_size_bias}, local_config_bias, mesh_device.get());
    }

    // ── Circular buffers ──
    // in0 (CB0): double-buffered blocks of A
    uint32_t in0_cb_tiles = in0_block_num_tiles * 2;
    CircularBufferConfig cb_in0_config =
        CircularBufferConfig(in0_cb_tiles * single_tile_size, {{CBIndex::c_0, cb_data_format}})
            .set_page_size(CBIndex::c_0, single_tile_size);
    CreateCircularBuffer(program, core, cb_in0_config);

    // in1 (CB1): double-buffered blocks of B
    uint32_t in1_cb_tiles = in1_block_num_tiles * 2;
    CircularBufferConfig cb_in1_config =
        CircularBufferConfig(in1_cb_tiles * single_tile_size, {{CBIndex::c_1, cb_data_format}})
            .set_page_size(CBIndex::c_1, single_tile_size);
    CreateCircularBuffer(program, core, cb_in1_config);

    // out (CB16) and interm (CB24) share L1 address space.
    // The FIFO management ensures they don't conflict (see design doc analysis).
    std::map<uint8_t, tt::DataFormat> partials_and_out_spec = {
        {CBIndex::c_24, cb_data_format}, {CBIndex::c_16, cb_data_format}};
    CircularBufferConfig cb_out_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, partials_and_out_spec)
            .set_page_size(CBIndex::c_24, single_tile_size)
            .set_page_size(CBIndex::c_16, single_tile_size);
    CreateCircularBuffer(program, core, cb_out_config);

    // bias (CB2): only when fusing bias
    if (cfg.fuse_bias) {
        CircularBufferConfig cb_bias_config =
            CircularBufferConfig(Nt * single_tile_size, {{CBIndex::c_2, cb_data_format}})
                .set_page_size(CBIndex::c_2, single_tile_size);
        CreateCircularBuffer(program, core, cb_bias_config);
    }

    // ── Reader kernel ──
    // Use the bias-capable reader (handles both bias and non-bias via runtime flag)
    auto reader_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_with_bias_blocked.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    // ── Writer kernel ──
    auto writer_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unswizzle.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // ── Compute kernel: parameterized via compile args + defines ──
    std::vector<uint32_t> compute_args = {
        in0_block_w,             // [0]
        in0_num_subblocks,       // [1]
        in0_block_num_tiles,     // [2]
        in0_subblock_num_tiles,  // [3]
        in1_num_subblocks,       // [4]
        in1_block_num_tiles,     // [5]
        in1_per_core_w,          // [6]
        num_blocks,              // [7]
        out_subblock_h,          // [8]
        out_subblock_w,          // [9]
        out_subblock_num_tiles,  // [10]
        1                        // [11] batch
    };

    std::map<std::string, std::string> defines;
    if (cfg.packer_l1_acc) {
        defines["PACKER_L1_ACC"] = "1";
    }
    if (cfg.pack_relu) {
        defines["PACK_RELU"] = "1";
    }
    if (cfg.fuse_bias) {
        defines["FUSE_BIAS"] = "1";
    }

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/test_matmul_helper_features_compute.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_args, .defines = defines});

    // ── Reader runtime args ──
    uint32_t in0_block_size_bytes = in0_block_num_tiles * single_tile_size;
    uint32_t in1_block_size_bytes = in1_block_num_tiles * single_tile_size;
    uint32_t bias_block_size_bytes = Nt * single_tile_size;

    std::vector<uint32_t> reader_args = {
        src0_dram->address(),
        0,  // bank_id
        src1_dram->address(),
        0,  // bank_id
        num_blocks,
        in0_block_num_tiles,
        in1_block_num_tiles,
        in0_block_size_bytes,
        in1_block_size_bytes,
        cfg.fuse_bias ? 1u : 0u,  // with_bias flag
    };
    if (cfg.fuse_bias) {
        reader_args.push_back(bias_dram->address());
        reader_args.push_back(0);  // bank_id
        reader_args.push_back(Nt);
        reader_args.push_back(bias_block_size_bytes);
    }
    SetRuntimeArgs(program, reader_id, core, reader_args);

    // ── Writer runtime args ──
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
         in0_num_subblocks,
         in1_num_subblocks,
         stride_r,
         stride_subblock_r,
         stride_subblock_c});

    // ── Generate random input data ──
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

    // Generate bias data (1D vector of N values, replicated across TILE_HEIGHT rows for tilizing)
    std::vector<bfloat16> bias_1d(cfg.N, bfloat16(0.0f));
    if (cfg.fuse_bias) {
        for (auto& v : bias_1d) {
            v = bfloat16(dist(rng));
        }
    }

    // ── CPU golden reference ──
    std::vector<bfloat16> golden_vec(cfg.M * cfg.N, bfloat16(0.0f));
    golden_matmul(src0_vec, src1_vec, bias_1d, golden_vec, cfg.M, cfg.N, cfg.K, cfg.fuse_bias, cfg.pack_relu);

    // ── Tilize inputs ──
    auto src0_tilized = tilize_nfaces(src0_vec, cfg.M, cfg.K);
    auto src1_tilized = tilize_nfaces(src1_vec, cfg.K, cfg.N);

    // Rearrange A to block-column order for the blocked reader
    src0_tilized = reorder_tiles_to_block_column(src0_tilized, Mt, Kt, in0_block_w);

    // Tilize bias: create [TILE_HEIGHT, N] matrix with bias replicated across all rows
    std::vector<bfloat16> bias_tilized;
    if (cfg.fuse_bias) {
        std::vector<bfloat16> bias_2d(TILE_HEIGHT * cfg.N);
        for (uint32_t row = 0; row < TILE_HEIGHT; row++) {
            for (uint32_t col = 0; col < cfg.N; col++) {
                bias_2d[row * cfg.N + col] = bias_1d[col];
            }
        }
        bias_tilized = tilize_nfaces(bias_2d, TILE_HEIGHT, cfg.N);
    }

    // ── Upload data ──
    auto src0_packed = pack_bfloat16_vec_into_uint32_vec(src0_tilized);
    auto src1_packed = pack_bfloat16_vec_into_uint32_vec(src1_tilized);
    fixture->WriteBuffer(mesh_device, src0_dram, src0_packed);
    fixture->WriteBuffer(mesh_device, src1_dram, src1_packed);

    if (cfg.fuse_bias) {
        auto bias_packed = pack_bfloat16_vec_into_uint32_vec(bias_tilized);
        fixture->WriteBuffer(mesh_device, bias_dram, bias_packed);
    }

    // ── Execute ──
    workload.add_program(device_range, std::move(program));
    fixture->RunProgram(mesh_device, workload);

    // ── Read back result ──
    std::vector<uint32_t> result_packed;
    fixture->ReadBuffer(mesh_device, dst_dram, result_packed);
    auto result_vec = unpack_uint32_vec_into_bfloat16_vec(result_packed);

    // Untilize result
    result_vec = untilize_nfaces(result_vec, cfg.M, cfg.N);

    float pcc = pcc_bfloat16(golden_vec, result_vec);

    // Build feature string for logging
    std::string features;
    if (cfg.packer_l1_acc) {
        features += "L1_ACC ";
    }
    if (cfg.pack_relu) {
        features += "RELU ";
    }
    if (cfg.fuse_bias) {
        features += "BIAS ";
    }
    if (features.empty()) {
        features = "basic ";
    }

    log_info(
        LogTest,
        "M={}, N={}, K={}, block_w={}, sub_h={}, sub_w={}, num_blocks={}, features=[{}] — PCC = {:.6f} (threshold: "
        "{})",
        cfg.M,
        cfg.N,
        cfg.K,
        cfg.in0_block_w,
        cfg.out_subblock_h,
        cfg.out_subblock_w,
        num_blocks,
        features,
        pcc,
        pcc_threshold);

    return pcc > pcc_threshold;
}

}  // namespace test_matmul_helper_features

using test_matmul_helper_features::run_matmul_helper_feature_test;

// ═════════════════════════════════════════════════════════════════════════════
// Test cases — each exercises a specific feature dimension from the Phase 2
// API design. Feature combinations match production kernel code paths.
// ═════════════════════════════════════════════════════════════════════════════

// ── L1 accumulation (PACKER_L1_ACC) ──────────────────────────────────────────
// Production kernel's primary optimization: avoids software spill/reload.
// With num_blocks=1, L1_ACC is a no-op (only one block, nothing to accumulate).
// With num_blocks>1, the hardware L1 accumulation path is exercised.

TEST_F(MeshDispatchFixture, TensixMatmulHelperL1AccSingleBlock) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_helper_feature_test(
            this,
            device,
            {.M = 64,
             .N = 64,
             .K = 64,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .in0_block_w = 2,
             .packer_l1_acc = true,
             .pack_relu = false,
             .fuse_bias = false}));
    }
}

TEST_F(MeshDispatchFixture, TensixMatmulHelperL1AccMultiBlock) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_helper_feature_test(
            this,
            device,
            {.M = 64,
             .N = 64,
             .K = 128,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .in0_block_w = 2,
             .packer_l1_acc = true,
             .pack_relu = false,
             .fuse_bias = false}));
    }
}

// ── PACK_RELU (no bias) ─────────────────────────────────────────────────────
// RELU applied during the pack phase on the last K-block's output.

TEST_F(MeshDispatchFixture, TensixMatmulHelperPackRelu) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_helper_feature_test(
            this,
            device,
            {.M = 64,
             .N = 64,
             .K = 64,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .in0_block_w = 2,
             .packer_l1_acc = false,
             .pack_relu = true,
             .fuse_bias = false}));
    }
}

TEST_F(MeshDispatchFixture, TensixMatmulHelperPackReluMultiBlock) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_helper_feature_test(
            this,
            device,
            {.M = 64,
             .N = 64,
             .K = 128,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .in0_block_w = 2,
             .packer_l1_acc = false,
             .pack_relu = true,
             .fuse_bias = false}));
    }
}

// ── Fused bias (add_bias_bcast_rows) ─────────────────────────────────────────
// matmul_block packs to interm_cb, then add_bias_bcast_rows adds row-broadcast
// bias and packs to out_cb.

TEST_F(MeshDispatchFixture, TensixMatmulHelperFusedBias) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_helper_feature_test(
            this,
            device,
            {.M = 64,
             .N = 64,
             .K = 64,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .in0_block_w = 2,
             .packer_l1_acc = false,
             .pack_relu = false,
             .fuse_bias = true}));
    }
}

TEST_F(MeshDispatchFixture, TensixMatmulHelperFusedBiasMultiBlock) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_helper_feature_test(
            this,
            device,
            {.M = 64,
             .N = 64,
             .K = 128,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .in0_block_w = 2,
             .packer_l1_acc = false,
             .pack_relu = false,
             .fuse_bias = true}));
    }
}

// ── L1_ACC + Bias ────────────────────────────────────────────────────────────
// Most common production path: L1 accumulation across K-blocks, then bias add.
// Uses packer_l1_acc=true + pack_last_to_interm=true (no reload, L1 accumulates
// all K-blocks, final result goes to interm for the bias phase).

TEST_F(MeshDispatchFixture, TensixMatmulHelperL1AccBias) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_helper_feature_test(
            this,
            device,
            {.M = 64,
             .N = 64,
             .K = 128,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .in0_block_w = 2,
             .packer_l1_acc = true,
             .pack_relu = false,
             .fuse_bias = true}));
    }
}

// ── Bias + RELU ──────────────────────────────────────────────────────────────
// RELU is applied during the bias phase's pack (not during matmul pack).
// The test kernel enables RELU between the matmul and bias helper calls.

TEST_F(MeshDispatchFixture, TensixMatmulHelperBiasRelu) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_helper_feature_test(
            this,
            device,
            {.M = 64,
             .N = 64,
             .K = 64,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .in0_block_w = 2,
             .packer_l1_acc = false,
             .pack_relu = true,
             .fuse_bias = true}));
    }
}

// ── L1_ACC + Bias + RELU (full production path) ─────────────────────────────
// Exercises the complete production kernel's most common path:
// PACKER_L1_ACC + FUSE_BIAS + PACK_RELU.

TEST_F(MeshDispatchFixture, TensixMatmulHelperL1AccBiasRelu) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_helper_feature_test(
            this,
            device,
            {.M = 64,
             .N = 64,
             .K = 128,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .in0_block_w = 2,
             .packer_l1_acc = true,
             .pack_relu = true,
             .fuse_bias = true}));
    }
}

// ── Multi-subblock stress tests ──────────────────────────────────────────────
// Larger matrices with multiple sub-blocks along both M and N dimensions.
// Validates that helpers iterate over sub-blocks correctly.

TEST_F(MeshDispatchFixture, TensixMatmulHelperMultiSubblockBias) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_helper_feature_test(
            this,
            device,
            {.M = 128,
             .N = 128,
             .K = 64,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .in0_block_w = 2,
             .packer_l1_acc = false,
             .pack_relu = false,
             .fuse_bias = true}));
    }
}

TEST_F(MeshDispatchFixture, TensixMatmulHelperMultiSubblockL1AccBias) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_helper_feature_test(
            this,
            device,
            {.M = 128,
             .N = 128,
             .K = 128,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .in0_block_w = 2,
             .packer_l1_acc = true,
             .pack_relu = false,
             .fuse_bias = true}));
    }
}

// ── L1_ACC + RELU (no bias) ─────────────────────────────────────────────────
// L1 accumulation with RELU applied on final pack (no bias phase).

TEST_F(MeshDispatchFixture, TensixMatmulHelperL1AccRelu) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_helper_feature_test(
            this,
            device,
            {.M = 64,
             .N = 64,
             .K = 128,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .in0_block_w = 2,
             .packer_l1_acc = true,
             .pack_relu = true,
             .fuse_bias = false}));
    }
}

}  // namespace tt::tt_metal
