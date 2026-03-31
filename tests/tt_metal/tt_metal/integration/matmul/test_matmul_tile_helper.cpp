// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Isolated integration tests for the matmul_tile helper (matmul_tile_helpers.hpp).
// Validates the tile-by-tile matmul pattern against a CPU golden reference.
//
// Phase 4, Instance 1 output.

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

namespace test_matmul_tile_helper {

// -- PCC (Pearson Correlation Coefficient) --

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

// -- Golden reference: simple matmul --

static void golden_matmul(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    uint32_t batch) {
    for (uint32_t nb = 0; nb < batch; nb++) {
        for (uint32_t i = 0; i < M; i++) {
            for (uint32_t j = 0; j < N; j++) {
                float acc = 0.0f;
                for (uint32_t k = 0; k < K; k++) {
                    acc +=
                        static_cast<float>(a[nb * M * K + i * K + k]) * static_cast<float>(b[nb * K * N + k * N + j]);
                }
                output[nb * M * N + i * N + j] = bfloat16(acc);
            }
        }
    }
}

// -- Rearrange tilized A and B into the tile-by-tile reader access order --
// The matmul_tile helper processes: for each (batch, mt, nt, kt), pop 1 A tile + 1 B tile.
// The reader reads tiles sequentially from DRAM, so we duplicate tiles as needed.
// A_dram[idx] = A_tile[b*Mt*Kt + mt*Kt + kt]  for idx at (b, mt, nt, kt)
// B_dram[idx] = B_tile[b*Kt*Nt + kt*Nt + nt]  for idx at (b, mt, nt, kt)

static std::vector<bfloat16> rearrange_a_for_tile_reader(
    const std::vector<bfloat16>& tilized_a, uint32_t Mt, uint32_t Nt, uint32_t Kt, uint32_t batch) {
    const uint32_t tile_elems = TILE_HEIGHT * TILE_WIDTH;
    uint32_t total_reads = batch * Mt * Nt * Kt;
    std::vector<bfloat16> result(total_reads * tile_elems);

    uint32_t dst = 0;
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t mt = 0; mt < Mt; mt++) {
            for (uint32_t nt = 0; nt < Nt; nt++) {
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    uint32_t src_tile = b * Mt * Kt + mt * Kt + kt;
                    std::copy(
                        tilized_a.begin() + src_tile * tile_elems,
                        tilized_a.begin() + (src_tile + 1) * tile_elems,
                        result.begin() + dst * tile_elems);
                    dst++;
                }
            }
        }
    }
    return result;
}

static std::vector<bfloat16> rearrange_b_for_tile_reader(
    const std::vector<bfloat16>& tilized_b, uint32_t Mt, uint32_t Nt, uint32_t Kt, uint32_t batch) {
    const uint32_t tile_elems = TILE_HEIGHT * TILE_WIDTH;
    uint32_t total_reads = batch * Mt * Nt * Kt;
    std::vector<bfloat16> result(total_reads * tile_elems);

    uint32_t dst = 0;
    for (uint32_t b = 0; b < batch; b++) {
        for (uint32_t mt = 0; mt < Mt; mt++) {
            for (uint32_t nt = 0; nt < Nt; nt++) {
                for (uint32_t kt = 0; kt < Kt; kt++) {
                    uint32_t src_tile = b * Kt * Nt + kt * Nt + nt;
                    std::copy(
                        tilized_b.begin() + src_tile * tile_elems,
                        tilized_b.begin() + (src_tile + 1) * tile_elems,
                        result.begin() + dst * tile_elems);
                    dst++;
                }
            }
        }
    }
    return result;
}

// -- Test configuration --

struct TileMatmulConfig {
    uint32_t M;      // rows (elements, multiple of 32)
    uint32_t N;      // cols (elements, multiple of 32)
    uint32_t K;      // inner dim (elements, multiple of 32)
    uint32_t batch;  // batch count
};

// -- Main test runner --

static bool run_matmul_tile_helper_test(
    MeshDispatchFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const TileMatmulConfig& cfg,
    float pcc_threshold = 0.97f) {
    uint32_t Mt = cfg.M / TILE_HEIGHT;
    uint32_t Nt = cfg.N / TILE_WIDTH;
    uint32_t Kt = cfg.K / TILE_WIDTH;
    uint32_t batch = cfg.batch;

    // Total tiles read/written
    uint32_t total_input_reads = batch * Mt * Nt * Kt;  // per-tile read count for each input
    uint32_t total_output_tiles = batch * Mt * Nt;

    uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    CoreCoord core({0, 0});

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program{};

    // -- DRAM buffers --
    uint32_t dram_size_a = single_tile_size * total_input_reads;
    uint32_t dram_size_b = single_tile_size * total_input_reads;
    uint32_t dram_size_c = single_tile_size * total_output_tiles;

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

    // -- Circular buffers --
    // in0 (CB0): double-buffered single tiles
    uint32_t in0_cb_tiles = 2;
    CircularBufferConfig cb_in0_config =
        CircularBufferConfig(in0_cb_tiles * single_tile_size, {{CBIndex::c_0, cb_data_format}})
            .set_page_size(CBIndex::c_0, single_tile_size);
    CreateCircularBuffer(program, core, cb_in0_config);

    // in1 (CB1): double-buffered single tiles
    uint32_t in1_cb_tiles = 2;
    CircularBufferConfig cb_in1_config =
        CircularBufferConfig(in1_cb_tiles * single_tile_size, {{CBIndex::c_1, cb_data_format}})
            .set_page_size(CBIndex::c_1, single_tile_size);
    CreateCircularBuffer(program, core, cb_in1_config);

    // out (CB16): single tile output buffer
    CircularBufferConfig cb_out_config = CircularBufferConfig(2 * single_tile_size, {{CBIndex::c_16, cb_data_format}})
                                             .set_page_size(CBIndex::c_16, single_tile_size);
    CreateCircularBuffer(program, core, cb_out_config);

    // -- Reader kernel --
    auto reader_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_small_block.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    // -- Writer kernel --
    auto writer_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_matmul_tile_sequential.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // -- Compute kernel --
    std::vector<uint32_t> compute_args = {Mt, Nt, Kt, batch};

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/test_matmul_tile_helper_compute.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_args});

    // -- Runtime args --
    SetRuntimeArgs(
        program,
        reader_id,
        core,
        {src0_dram->address(),
         0,  // bank_id
         src1_dram->address(),
         0,                    // bank_id
         total_input_reads});  // num_tiles (pairs of tiles to read)

    SetRuntimeArgs(
        program,
        writer_id,
        core,
        {dst_dram->address(),
         0,  // bank_id
         total_output_tiles});

    // -- Generate random input data --
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    uint32_t total_a_elems = batch * cfg.M * cfg.K;
    uint32_t total_b_elems = batch * cfg.K * cfg.N;
    uint32_t total_c_elems = batch * cfg.M * cfg.N;

    std::vector<bfloat16> src0_vec(total_a_elems);
    std::vector<bfloat16> src1_vec(total_b_elems);
    for (auto& v : src0_vec) {
        v = bfloat16(dist(rng));
    }
    for (auto& v : src1_vec) {
        v = bfloat16(dist(rng));
    }

    // -- CPU golden reference --
    std::vector<bfloat16> golden_vec(total_c_elems, bfloat16(0.0f));
    golden_matmul(src0_vec, src1_vec, golden_vec, cfg.M, cfg.N, cfg.K, batch);

    // -- Tilize inputs (per-batch) --
    // Tilize each batch's A and B matrices separately, then concatenate.
    std::vector<bfloat16> src0_tilized_all;
    std::vector<bfloat16> src1_tilized_all;
    for (uint32_t b = 0; b < batch; b++) {
        std::vector<bfloat16> a_batch(src0_vec.begin() + b * cfg.M * cfg.K, src0_vec.begin() + (b + 1) * cfg.M * cfg.K);
        std::vector<bfloat16> b_batch(src1_vec.begin() + b * cfg.K * cfg.N, src1_vec.begin() + (b + 1) * cfg.K * cfg.N);

        auto a_tilized = tilize_nfaces(a_batch, cfg.M, cfg.K);
        auto b_tilized = tilize_nfaces(b_batch, cfg.K, cfg.N);

        src0_tilized_all.insert(src0_tilized_all.end(), a_tilized.begin(), a_tilized.end());
        src1_tilized_all.insert(src1_tilized_all.end(), b_tilized.begin(), b_tilized.end());
    }

    // -- Rearrange for reader access order --
    auto src0_reader = rearrange_a_for_tile_reader(src0_tilized_all, Mt, Nt, Kt, batch);
    auto src1_reader = rearrange_b_for_tile_reader(src1_tilized_all, Mt, Nt, Kt, batch);

    // -- Upload data --
    auto src0_packed = pack_bfloat16_vec_into_uint32_vec(src0_reader);
    auto src1_packed = pack_bfloat16_vec_into_uint32_vec(src1_reader);
    fixture->WriteBuffer(mesh_device, src0_dram, src0_packed);
    fixture->WriteBuffer(mesh_device, src1_dram, src1_packed);

    // -- Execute --
    workload.add_program(device_range, std::move(program));
    fixture->RunProgram(mesh_device, workload);

    // -- Read back result --
    std::vector<uint32_t> result_packed;
    fixture->ReadBuffer(mesh_device, dst_dram, result_packed);
    auto result_vec = unpack_uint32_vec_into_bfloat16_vec(result_packed);

    // -- Untilize result (per-batch) --
    std::vector<bfloat16> result_untilized;
    uint32_t tiles_per_batch_out = Mt * Nt * TILE_HEIGHT * TILE_WIDTH;
    for (uint32_t b = 0; b < batch; b++) {
        std::vector<bfloat16> batch_tilized(
            result_vec.begin() + b * tiles_per_batch_out, result_vec.begin() + (b + 1) * tiles_per_batch_out);
        auto batch_untilized = untilize_nfaces(batch_tilized, cfg.M, cfg.N);
        result_untilized.insert(result_untilized.end(), batch_untilized.begin(), batch_untilized.end());
    }

    float pcc = pcc_bfloat16(golden_vec, result_untilized);

    log_info(
        LogTest,
        "M={}, N={}, K={}, batch={} — PCC = {:.6f} (threshold: {})",
        cfg.M,
        cfg.N,
        cfg.K,
        batch,
        pcc,
        pcc_threshold);

    return pcc > pcc_threshold;
}

}  // namespace test_matmul_tile_helper

using test_matmul_tile_helper::run_matmul_tile_helper_test;

// =========================================================================
// Test cases — exercise the matmul_tile helper with various configurations.
// =========================================================================

// -- Basic: single output tile (Mt=1, Nt=1, Kt=1) --
TEST_F(MeshDispatchFixture, TensixMatmulTileHelperBasic1x1x1) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_tile_helper_test(this, device, {.M = 32, .N = 32, .K = 32, .batch = 1}));
    }
}

// -- K-dimension accumulation (Kt > 1) --
TEST_F(MeshDispatchFixture, TensixMatmulTileHelperKAccum) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_tile_helper_test(this, device, {.M = 32, .N = 32, .K = 128, .batch = 1}));
    }
}

// -- Multiple output tiles (Mt > 1, Nt > 1) --
TEST_F(MeshDispatchFixture, TensixMatmulTileHelperMultiTile) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_tile_helper_test(this, device, {.M = 64, .N = 64, .K = 64, .batch = 1}));
    }
}

// -- Non-square dimensions --
TEST_F(MeshDispatchFixture, TensixMatmulTileHelperNonSquare) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_tile_helper_test(this, device, {.M = 32, .N = 64, .K = 96, .batch = 1}));
    }
}

// -- Batch support --
TEST_F(MeshDispatchFixture, TensixMatmulTileHelperBatch) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_tile_helper_test(this, device, {.M = 32, .N = 32, .K = 64, .batch = 2}));
    }
}

// -- Larger test with batch --
TEST_F(MeshDispatchFixture, TensixMatmulTileHelperLargerBatch) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_tile_helper_test(this, device, {.M = 64, .N = 32, .K = 64, .batch = 3}));
    }
}

}  // namespace tt::tt_metal
