// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Isolated integration tests for matmul_reduce_inplace
// (ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp).
//
// The helper reduces `in_out_cb` in place by matmul-ing against a
// column-identity tile in `in1_cb`. Net CB tile count is unchanged: for
// every subblock_h tiles consumed, subblock_h tiles are pushed back.
//
// Test strategy:
//   - Seed in_out_cb with an M_tiles × 1-tile column of random bf16 data
//     (so each "tile row" holds 32 rows × 32 cols of test data).
//   - Seed in1_cb with a column-identity tile (value=1.0 at column 0 of
//     every tile row, zeros elsewhere) — matches SDPA's usage produced by
//     generate_bcast_col_scalar on the reader side.
//   - After the reduce, column 0 of each output tile holds the row-sum of
//     the corresponding input tile row. Other columns are zero.
//   - Also asserts the in-place invariant: total tiles in c_0 unchanged.
//
// Varies STATS_GRANULARITY (= subblock_h) and M to walk the num_subblocks
// dimension. block_kt and subblock_w stay at 1 (SDPA usage).

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

namespace test_matmul_reduce_inplace {

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

struct ReduceConfig {
    // M (in elements) = num_subblocks * subblock_h * TILE_HEIGHT
    uint32_t num_subblocks;
    uint32_t subblock_h;
    // Kept at 1 per SDPA usage.
    uint32_t subblock_w = 1;
    uint32_t block_kt = 1;
};

static bool run_matmul_reduce_inplace_test(
    MeshDispatchFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const ReduceConfig& cfg,
    float pcc_threshold = 0.9999f) {
    const uint32_t M_tiles = cfg.num_subblocks * cfg.subblock_h;
    const uint32_t N_tiles = cfg.subblock_w;
    const uint32_t total_tiles = M_tiles * N_tiles;
    const uint32_t M = M_tiles * TILE_HEIGHT;
    const uint32_t N = N_tiles * TILE_WIDTH;
    const uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;

    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    CoreCoord core({0, 0});

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program{};

    // DRAM buffers: input, col-ident (one tile), output copy.
    uint32_t dram_size_in = single_tile_size * total_tiles;
    uint32_t dram_size_cid = single_tile_size;
    uint32_t dram_size_out = single_tile_size * total_tiles;

    distributed::DeviceLocalBufferConfig lc_in{
        .page_size = dram_size_in, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::DeviceLocalBufferConfig lc_cid{
        .page_size = dram_size_cid, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::DeviceLocalBufferConfig lc_out{
        .page_size = dram_size_out, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};

    auto in_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = dram_size_in}, lc_in, mesh_device.get());
    auto cid_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = dram_size_cid}, lc_cid, mesh_device.get());
    auto out_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = dram_size_out}, lc_out, mesh_device.get());

    // CBs:
    //   c_0  (in_out)   — compute-owned in/out CB for the reduce (double-buffered)
    //   c_1  (col-ident) — single column-identity tile
    //   c_2  (staging)  — reader-produced input; compute copies to c_0 to
    //                     establish compute ownership before matmul_reduce_inplace
    //   c_16 (out-copy) — reduced output for the writer to drain
    //
    // c_2 is necessary because matmul_reduce_inplace requires in_out_cb to be
    // compute-produced. If the reader fills c_0 directly, T2's tiles_received
    // shadow for c_0 starts at 0 while L1's counter is already 1; T2's push-back
    // then writes 1 to L1 (instead of 2), causing cb_wait_front to deadlock.
    uint32_t c0_tiles = total_tiles * 2;  // double-buffered
    CircularBufferConfig cb0_cfg = CircularBufferConfig(c0_tiles * single_tile_size, {{CBIndex::c_0, cb_data_format}})
                                       .set_page_size(CBIndex::c_0, single_tile_size);
    CreateCircularBuffer(program, core, cb0_cfg);

    CircularBufferConfig cb1_cfg = CircularBufferConfig(single_tile_size, {{CBIndex::c_1, cb_data_format}})
                                       .set_page_size(CBIndex::c_1, single_tile_size);
    CreateCircularBuffer(program, core, cb1_cfg);

    // c_2: reader staging CB (reader pushes here, compute consumes and copies to c_0)
    uint32_t c2_tiles = total_tiles * 2;
    CircularBufferConfig cb2_cfg = CircularBufferConfig(c2_tiles * single_tile_size, {{CBIndex::c_2, cb_data_format}})
                                       .set_page_size(CBIndex::c_2, single_tile_size);
    CreateCircularBuffer(program, core, cb2_cfg);

    uint32_t c16_tiles = total_tiles * 2;
    CircularBufferConfig cb16_cfg =
        CircularBufferConfig(c16_tiles * single_tile_size, {{CBIndex::c_16, cb_data_format}})
            .set_page_size(CBIndex::c_16, single_tile_size);
    CreateCircularBuffer(program, core, cb16_cfg);

    // Reader: feeds c_2 (staging, total_tiles) from DRAM then pushes col-ident into c_1.
    // We use a lightweight inline reader kernel — generate a custom one.
    auto reader_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_reduce_inplace.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    // Writer: streams c_16 to DRAM.
    auto writer_id = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    std::vector<uint32_t> compute_args = {cfg.num_subblocks, cfg.subblock_h, cfg.subblock_w, cfg.block_kt, total_tiles};
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/test_matmul_reduce_inplace_compute.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_args});

    SetRuntimeArgs(
        program,
        reader_id,
        core,
        {in_dram->address(), 0, cid_dram->address(), 0, total_tiles, total_tiles * single_tile_size, single_tile_size});

    SetRuntimeArgs(program, writer_id, core, {out_dram->address(), 0, total_tiles});

    // ── Input data ──
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Fill input as M × N (elements).
    std::vector<bfloat16> src_vec(M * N);
    for (auto& v : src_vec) {
        v = bfloat16(dist(rng));
    }

    // Col-ident tile: scalar 1.0 in column 0 of the tile, zeros elsewhere.
    // Mirrors generate_bcast_col_scalar on the device side.
    std::vector<bfloat16> cid_tile(TILE_HEIGHT * TILE_WIDTH, bfloat16(0.0f));
    for (uint32_t r = 0; r < TILE_HEIGHT; r++) {
        cid_tile[r * TILE_WIDTH + 0] = bfloat16(1.0f);
    }
    // Host-side "tilize" of a 32×32 tile requires face packing. The input
    // isn't a tiled "2D matrix" — it's a single tile. Use tilize_nfaces on
    // a 32×32 element matrix.
    auto cid_tilized = tilize_nfaces(cid_tile, TILE_HEIGHT, TILE_WIDTH);

    // ── Golden ──
    // For each tile of the input, the reduce computes col_0 = row_sum,
    // cols 1..N-1 = 0. Golden produces an M×N element matrix.
    //
    // NB: the "row sum" is the sum across the N=1 tile's 32 columns.
    // If we had multiple N tiles they would all sum together.
    std::vector<bfloat16> golden(M * N, bfloat16(0.0f));
    for (uint32_t i = 0; i < M; i++) {
        float row_sum = 0.0f;
        for (uint32_t j = 0; j < N; j++) {
            row_sum += static_cast<float>(src_vec[i * N + j]);
        }
        golden[i * N + 0] = bfloat16(row_sum);
        // cols >=1 stay zero.
    }

    auto src_tilized = tilize_nfaces(src_vec, M, N);

    auto src_packed = pack_bfloat16_vec_into_uint32_vec(src_tilized);
    auto cid_packed = pack_bfloat16_vec_into_uint32_vec(cid_tilized);
    fixture->WriteBuffer(mesh_device, in_dram, src_packed);
    fixture->WriteBuffer(mesh_device, cid_dram, cid_packed);

    workload.add_program(device_range, std::move(program));
    fixture->RunProgram(mesh_device, workload);

    // ── Read back ──
    // Expected: total_tiles were pushed into c_16 (post-reduce), writer drained.
    // That is the in-place invariant check: same number of tiles after reduce.
    std::vector<uint32_t> result_packed;
    fixture->ReadBuffer(mesh_device, out_dram, result_packed);
    auto result_tilized = unpack_uint32_vec_into_bfloat16_vec(result_packed);

    if (result_tilized.size() != M * N) {
        log_warning(
            LogTest, "Result size {} != expected {} — in-place invariant violated?", result_tilized.size(), M * N);
        return false;
    }

    auto result_vec = untilize_nfaces(result_tilized, M, N);

    float pcc = pcc_bfloat16(golden, result_vec);
    log_info(
        LogTest,
        "num_subblocks={} subblock_h={} subblock_w={} block_kt={} M_tiles={} N_tiles={} — PCC = {:.6f} (thresh {:.4f})",
        cfg.num_subblocks,
        cfg.subblock_h,
        cfg.subblock_w,
        cfg.block_kt,
        M_tiles,
        N_tiles,
        pcc,
        pcc_threshold);
    return pcc > pcc_threshold;
}

}  // namespace test_matmul_reduce_inplace

using test_matmul_reduce_inplace::ReduceConfig;
using test_matmul_reduce_inplace::run_matmul_reduce_inplace_test;

// STATS_GRANULARITY=1 (default SDPA decode path)
TEST_F(MeshDispatchFixture, TensixMatmulReduceInplaceGranularity1M1) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_reduce_inplace_test(this, device, {.num_subblocks = 1, .subblock_h = 1}));
    }
}

TEST_F(MeshDispatchFixture, TensixMatmulReduceInplaceGranularity1M4) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_reduce_inplace_test(this, device, {.num_subblocks = 4, .subblock_h = 1}));
    }
}

TEST_F(MeshDispatchFixture, TensixMatmulReduceInplaceGranularity1M8) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_reduce_inplace_test(this, device, {.num_subblocks = 8, .subblock_h = 1}));
    }
}

// STATS_GRANULARITY=2 (chunked decode path)
TEST_F(MeshDispatchFixture, TensixMatmulReduceInplaceGranularity2) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_reduce_inplace_test(this, device, {.num_subblocks = 4, .subblock_h = 2}));
    }
}

// STATS_GRANULARITY=4 (larger chunk)
TEST_F(MeshDispatchFixture, TensixMatmulReduceInplaceGranularity4) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_reduce_inplace_test(this, device, {.num_subblocks = 2, .subblock_h = 4}));
    }
}

}  // namespace tt::tt_metal
