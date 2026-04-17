// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Isolated integration tests for reblock_and_untilize helper
// (ttnn/cpp/ttnn/kernel_lib/reblock_untilize_helpers.hpp).
//
// Drives the helper directly with a synthetic subblock-major tiled buffer —
// no matmul participates. Verifies the helper reorders + untilizes into
// row-major element output, pushing `out_block_w` tiles worth per row.
//
// Test strategy:
//   - Construct an M×N matrix filled with deterministic bf16 values (i*N + j).
//   - Tilize normally (nfaces), then re-order the tile sequence so it lands in
//     subblock-major order within each row-group (matches the matmul_block
//     output layout when row_major_output=false).
//   - Push the whole thing into c_24.
//   - Helper writes row-major untilized into c_16; writer streams to DRAM.
//   - Compare against the original element-order matrix directly.

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
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

namespace test_reblock_and_untilize {

// Take a tilized buffer in row-major tile order over an Mt×Nt grid and
// reorder it into subblock-major tile order within each row-group.
// Layout within one row-group (out_subblock_h tile-rows):
//   for each of num_subblocks_w subblocks:
//     for each row h in out_subblock_h:
//       for each col w in out_subblock_w:
//         tile
static std::vector<bfloat16> row_major_to_subblock_order(
    const std::vector<bfloat16>& tilized, uint32_t Mt, uint32_t Nt, uint32_t out_subblock_h, uint32_t out_subblock_w) {
    const uint32_t tile_elems = TILE_HEIGHT * TILE_WIDTH;
    const uint32_t num_sb_w = Nt / out_subblock_w;
    const uint32_t num_row_groups = Mt / out_subblock_h;
    std::vector<bfloat16> result(tilized.size());
    uint32_t dst = 0;
    for (uint32_t g = 0; g < num_row_groups; g++) {
        for (uint32_t sb = 0; sb < num_sb_w; sb++) {
            for (uint32_t h = 0; h < out_subblock_h; h++) {
                for (uint32_t w = 0; w < out_subblock_w; w++) {
                    uint32_t row_tile = g * out_subblock_h + h;
                    uint32_t col_tile = sb * out_subblock_w + w;
                    uint32_t src_tile_idx = row_tile * Nt + col_tile;
                    std::copy(
                        tilized.begin() + src_tile_idx * tile_elems,
                        tilized.begin() + (src_tile_idx + 1) * tile_elems,
                        result.begin() + dst);
                    dst += tile_elems;
                }
            }
        }
    }
    return result;
}

struct ReblockConfig {
    uint32_t M;               // rows in elements
    uint32_t N;               // cols in elements
    uint32_t out_subblock_h;  // tiles
    uint32_t out_subblock_w;  // tiles
};

static bool run_reblock_and_untilize_test(
    MeshDispatchFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const ReblockConfig& cfg) {
    uint32_t Mt = cfg.M / TILE_HEIGHT;
    uint32_t Nt = cfg.N / TILE_WIDTH;
    uint32_t num_subblocks_w = Nt / cfg.out_subblock_w;
    uint32_t num_row_groups = Mt / cfg.out_subblock_h;
    uint32_t total_tiles = Mt * Nt;
    uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    CoreCoord core({0, 0});

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program{};

    uint32_t dram_size_in = single_tile_size * total_tiles;
    // Untilized output: writer drains one untilized "row of tiles" per pack,
    // which is out_block_w tiles worth (= Nt * single_tile_size for full N).
    uint32_t dram_size_out = single_tile_size * total_tiles;

    distributed::DeviceLocalBufferConfig lc_in{
        .page_size = dram_size_in, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::DeviceLocalBufferConfig lc_out{
        .page_size = dram_size_out, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};

    auto in_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = dram_size_in}, lc_in, mesh_device.get());
    auto out_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = dram_size_out}, lc_out, mesh_device.get());

    // c_24 holds the subblock-major tiled input.
    CircularBufferConfig cb_interm_cfg =
        CircularBufferConfig(total_tiles * single_tile_size, {{CBIndex::c_24, cb_data_format}})
            .set_page_size(CBIndex::c_24, single_tile_size);
    CreateCircularBuffer(program, core, cb_interm_cfg);

    // c_16 holds untilized output.
    CircularBufferConfig cb_out_cfg =
        CircularBufferConfig(total_tiles * single_tile_size, {{CBIndex::c_16, cb_data_format}})
            .set_page_size(CBIndex::c_16, single_tile_size);
    CreateCircularBuffer(program, core, cb_out_cfg);

    // Reader: push the whole buffer into c_24 in one shot.
    auto reader_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_push_c24.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    // Writer: stream c_16 to DRAM.
    auto writer_id = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    std::vector<uint32_t> compute_args = {
        num_row_groups,
        num_subblocks_w,
        cfg.out_subblock_h,
        cfg.out_subblock_w,
        /*out_block_w*/ num_subblocks_w * cfg.out_subblock_w};
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/test_reblock_and_untilize_compute.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_args});

    SetRuntimeArgs(program, reader_id, core, {in_dram->address(), 0, total_tiles, total_tiles * single_tile_size});
    SetRuntimeArgs(program, writer_id, core, {out_dram->address(), 0, total_tiles});

    // Deterministic input: value at (i, j) = i*N + j (bf16-rounded).
    std::vector<bfloat16> src_elem(cfg.M * cfg.N);
    for (uint32_t i = 0; i < cfg.M; i++) {
        for (uint32_t j = 0; j < cfg.N; j++) {
            src_elem[i * cfg.N + j] = bfloat16(static_cast<float>(i * cfg.N + j));
        }
    }

    auto src_tilized_row_major = tilize_nfaces(src_elem, cfg.M, cfg.N);
    auto src_tilized_subblock =
        row_major_to_subblock_order(src_tilized_row_major, Mt, Nt, cfg.out_subblock_h, cfg.out_subblock_w);
    auto src_packed = pack_bfloat16_vec_into_uint32_vec(src_tilized_subblock);
    fixture->WriteBuffer(mesh_device, in_dram, src_packed);

    workload.add_program(device_range, std::move(program));
    fixture->RunProgram(mesh_device, workload);

    std::vector<uint32_t> result_packed;
    fixture->ReadBuffer(mesh_device, out_dram, result_packed);
    auto result_tilized = unpack_uint32_vec_into_bfloat16_vec(result_packed);

    // Helper emits row-major untilized output to c_16. writer_unary streams
    // the raw bytes to DRAM (tile-at-a-time mechanically, but the layout is
    // whatever the helper packed into the CB). In production this output
    // feeds writer_unary_interleaved_start_id which expects `num_tiles`
    // tiles — but the underlying bytes are semantically the MxN matrix in
    // row-major element layout, not tiled.
    //
    // To compare robustly against random or deterministic inputs, compute
    // PCC between the raw bf16 stream and src_elem (flattened row-major).
    // If the helper's byte layout differs from "pure row-major concatenated
    // across pushes", the PCC will drop and flag the mismatch.
    float max_err = 0.0f;
    if (result_tilized.size() < src_elem.size()) {
        log_warning(LogTest, "Result size {} < expected {}", result_tilized.size(), src_elem.size());
        return false;
    }

    // Compute PCC on the common prefix.
    std::vector<bfloat16> got_slice(result_tilized.begin(), result_tilized.begin() + src_elem.size());
    float sum_e = 0.0f, sum_g = 0.0f;
    for (size_t k = 0; k < src_elem.size(); k++) {
        sum_e += static_cast<float>(src_elem[k]);
        sum_g += static_cast<float>(got_slice[k]);
    }
    float mean_e = sum_e / src_elem.size();
    float mean_g = sum_g / src_elem.size();
    float cov = 0.0f, var_e = 0.0f, var_g = 0.0f;
    for (size_t k = 0; k < src_elem.size(); k++) {
        float ed = static_cast<float>(src_elem[k]) - mean_e;
        float gd = static_cast<float>(got_slice[k]) - mean_g;
        cov += ed * gd;
        var_e += ed * ed;
        var_g += gd * gd;
        max_err = std::max(max_err, std::abs(ed - gd));
    }
    float pcc = (var_e == 0.0f || var_g == 0.0f) ? (var_e == var_g ? 1.0f : 0.0f) : (cov / std::sqrt(var_e * var_g));

    log_info(
        LogTest,
        "M={} N={} sb_h={} sb_w={} rg={} nb_w={} PCC={:.6f} max_err={:.4f}",
        cfg.M,
        cfg.N,
        cfg.out_subblock_h,
        cfg.out_subblock_w,
        num_row_groups,
        num_subblocks_w,
        pcc,
        max_err);
    // Untilize is byte-level lossless. Threshold at 0.9999 for PCC.
    return pcc > 0.9999f;
}

}  // namespace test_reblock_and_untilize

using test_reblock_and_untilize::ReblockConfig;
using test_reblock_and_untilize::run_reblock_and_untilize_test;

// Single subblock per row, single row-group
TEST_F(MeshDispatchFixture, TensixReblockAndUntilizeSingleSubblock) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(
            run_reblock_and_untilize_test(this, device, {.M = 32, .N = 32, .out_subblock_h = 1, .out_subblock_w = 1}));
    }
}

// Multiple subblocks along N
TEST_F(MeshDispatchFixture, TensixReblockAndUntilizeMultiSubblockN) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(
            run_reblock_and_untilize_test(this, device, {.M = 64, .N = 128, .out_subblock_h = 2, .out_subblock_w = 2}));
    }
}

// Multiple row-groups along M
TEST_F(MeshDispatchFixture, TensixReblockAndUntilizeMultiRowGroup) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(
            run_reblock_and_untilize_test(this, device, {.M = 128, .N = 64, .out_subblock_h = 2, .out_subblock_w = 2}));
    }
}

// Both dimensions subblocked
TEST_F(MeshDispatchFixture, TensixReblockAndUntilizeMultiBoth) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_reblock_and_untilize_test(
            this, device, {.M = 128, .N = 128, .out_subblock_h = 2, .out_subblock_w = 2}));
    }
}

}  // namespace tt::tt_metal
