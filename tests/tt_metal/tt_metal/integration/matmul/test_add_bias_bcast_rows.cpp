// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Isolated integration tests for add_bias_bcast_rows helper
// (ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp).
//
// The helper reads matmul partials from partials_cb, adds a row-broadcast
// bias from bias_cb, and writes to out_cb. Caller owns bias CB wait/pop
// lifecycle. Tests cover the two production patterns:
//   (a) BIAS_ONE_TIME_FRONT: reader pushes bias once, helper called N times,
//       caller waits once on first iter and never pops (SDPA + non-chunked
//       matmul path where num_blocks_w_dim == 1).
//   (b) BIAS_PER_ITER_PUSH: reader pushes per-iter, caller waits+pops per-iter
//       (matmul path where num_blocks_w_dim > 1).
//   And both patterns with a nontrivial PostBiasFn (relu) to exercise the
//   functor slot.
//
// Inputs are synthetic (random partials + random bias, no upstream matmul).

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

namespace test_add_bias_bcast_rows {

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

// Re-tile a row-major-tile-order tilized buffer into subblock-major order
// (matches what matmul_block<row_major_output=false> produces in partials_cb).
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

// Reverse of the above: helper emits output in subblock order (reserve+push
// per subblock); we need to reassemble the tile stream back into row-major
// for comparison. Total data size matches.
static std::vector<bfloat16> subblock_order_to_row_major(
    const std::vector<bfloat16>& subblock_tilized,
    uint32_t Mt,
    uint32_t Nt,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w) {
    const uint32_t tile_elems = TILE_HEIGHT * TILE_WIDTH;
    const uint32_t num_sb_w = Nt / out_subblock_w;
    const uint32_t num_row_groups = Mt / out_subblock_h;
    std::vector<bfloat16> result(subblock_tilized.size());
    uint32_t src = 0;
    for (uint32_t g = 0; g < num_row_groups; g++) {
        for (uint32_t sb = 0; sb < num_sb_w; sb++) {
            for (uint32_t h = 0; h < out_subblock_h; h++) {
                for (uint32_t w = 0; w < out_subblock_w; w++) {
                    uint32_t row_tile = g * out_subblock_h + h;
                    uint32_t col_tile = sb * out_subblock_w + w;
                    uint32_t dst_tile_idx = row_tile * Nt + col_tile;
                    std::copy(
                        subblock_tilized.begin() + src,
                        subblock_tilized.begin() + src + tile_elems,
                        result.begin() + dst_tile_idx * tile_elems);
                    src += tile_elems;
                }
            }
        }
    }
    return result;
}

enum class BiasPattern { OneTimeFront, PerIterPush };

struct BiasConfig {
    uint32_t M;
    uint32_t N;
    uint32_t out_subblock_h;
    uint32_t out_subblock_w;
    uint32_t num_invocations;
    BiasPattern pattern;
    bool post_bias_relu = false;
};

static bool run_add_bias_bcast_rows_test(
    MeshDispatchFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const BiasConfig& cfg,
    float pcc_threshold = 0.9995f) {
    uint32_t Mt = cfg.M / TILE_HEIGHT;
    uint32_t Nt = cfg.N / TILE_WIDTH;
    uint32_t in0_num_subblocks = Mt / cfg.out_subblock_h;
    uint32_t in1_num_subblocks = Nt / cfg.out_subblock_w;
    uint32_t bias_ntiles = Nt;
    uint32_t partials_tiles_per_iter = Mt * Nt;
    uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    CoreCoord core({0, 0});

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program{};

    // DRAM: partials (num_invocations copies), bias (1 or num_invocations copies), output.
    uint32_t partials_bytes_total = single_tile_size * partials_tiles_per_iter * cfg.num_invocations;
    uint32_t bias_copies = (cfg.pattern == BiasPattern::OneTimeFront) ? 1u : cfg.num_invocations;
    uint32_t bias_bytes_total = single_tile_size * bias_ntiles * bias_copies;
    uint32_t out_bytes_total = partials_bytes_total;

    distributed::DeviceLocalBufferConfig lc_p{
        .page_size = partials_bytes_total, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::DeviceLocalBufferConfig lc_b{
        .page_size = bias_bytes_total, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};
    distributed::DeviceLocalBufferConfig lc_o{
        .page_size = out_bytes_total, .buffer_type = tt_metal::BufferType::DRAM, .bottom_up = false};

    auto partials_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = partials_bytes_total}, lc_p, mesh_device.get());
    auto bias_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = bias_bytes_total}, lc_b, mesh_device.get());
    auto out_dram = distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = out_bytes_total}, lc_o, mesh_device.get());

    // CBs. Size partials and out for one iter's worth; bias CB holds enough
    // tiles for one push cycle.
    CircularBufferConfig cb_partials_cfg =
        CircularBufferConfig(partials_tiles_per_iter * single_tile_size, {{CBIndex::c_24, cb_data_format}})
            .set_page_size(CBIndex::c_24, single_tile_size);
    CreateCircularBuffer(program, core, cb_partials_cfg);

    CircularBufferConfig cb_bias_cfg =
        CircularBufferConfig(bias_ntiles * single_tile_size, {{CBIndex::c_2, cb_data_format}})
            .set_page_size(CBIndex::c_2, single_tile_size);
    CreateCircularBuffer(program, core, cb_bias_cfg);

    CircularBufferConfig cb_out_cfg =
        CircularBufferConfig(partials_tiles_per_iter * single_tile_size, {{CBIndex::c_16, cb_data_format}})
            .set_page_size(CBIndex::c_16, single_tile_size);
    CreateCircularBuffer(program, core, cb_out_cfg);

    // Reader: pushes partials per-iter, bias once-or-per-iter per the pattern.
    auto reader_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_bias_test.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    auto writer_id = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    std::vector<uint32_t> compute_args = {
        cfg.num_invocations, in0_num_subblocks, in1_num_subblocks, cfg.out_subblock_h, cfg.out_subblock_w, bias_ntiles};

    std::map<std::string, std::string> defines;
    if (cfg.pattern == BiasPattern::OneTimeFront) {
        defines["BIAS_ONE_TIME_FRONT"] = "1";
    } else {
        defines["BIAS_PER_ITER_PUSH"] = "1";
    }
    if (cfg.post_bias_relu) {
        defines["HELPER_POST_BIAS_RELU"] = "1";
    }

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/test_add_bias_bcast_rows_compute.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_args, .defines = defines});

    uint32_t partials_bytes_per_iter = partials_tiles_per_iter * single_tile_size;
    uint32_t bias_bytes_per_push = bias_ntiles * single_tile_size;

    SetRuntimeArgs(
        program,
        reader_id,
        core,
        {partials_dram->address(),
         0,
         bias_dram->address(),
         0,
         cfg.num_invocations,
         partials_tiles_per_iter,
         bias_ntiles,
         partials_bytes_per_iter,
         bias_bytes_per_push,
         (cfg.pattern == BiasPattern::OneTimeFront) ? 1u : 0u});

    SetRuntimeArgs(program, writer_id, core, {out_dram->address(), 0, partials_tiles_per_iter * cfg.num_invocations});

    // ── Inputs ──
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(cfg.post_bias_relu ? -1.0f : 0.0f, 1.0f);

    // Partials: per iter, an MxN random matrix, pre-tilized in subblock order.
    std::vector<bfloat16> partials_all;
    std::vector<std::vector<bfloat16>> partials_per_iter_rowmajor;
    partials_per_iter_rowmajor.reserve(cfg.num_invocations);
    for (uint32_t i = 0; i < cfg.num_invocations; i++) {
        std::vector<bfloat16> p(cfg.M * cfg.N);
        for (auto& v : p) {
            v = bfloat16(dist(rng));
        }
        partials_per_iter_rowmajor.push_back(p);
        auto t = tilize_nfaces(p, cfg.M, cfg.N);
        auto sb = row_major_to_subblock_order(t, Mt, Nt, cfg.out_subblock_h, cfg.out_subblock_w);
        partials_all.insert(partials_all.end(), sb.begin(), sb.end());
    }

    // Bias: per-push, a 1-row bias repeated across tile height, tilized.
    std::vector<bfloat16> bias_all;
    std::vector<std::vector<bfloat16>> bias_per_push_1d;
    bias_per_push_1d.reserve(bias_copies);
    for (uint32_t p = 0; p < bias_copies; p++) {
        std::vector<bfloat16> b(cfg.N);
        for (auto& v : b) {
            v = bfloat16(dist(rng));
        }
        bias_per_push_1d.push_back(b);
        // Build 2D tile: [TILE_HEIGHT, N] with bias replicated down the rows.
        std::vector<bfloat16> b2d(TILE_HEIGHT * cfg.N);
        for (uint32_t r = 0; r < TILE_HEIGHT; r++) {
            for (uint32_t c = 0; c < cfg.N; c++) {
                b2d[r * cfg.N + c] = b[c];
            }
        }
        auto bt = tilize_nfaces(b2d, TILE_HEIGHT, cfg.N);
        bias_all.insert(bias_all.end(), bt.begin(), bt.end());
    }

    auto partials_packed = pack_bfloat16_vec_into_uint32_vec(partials_all);
    auto bias_packed = pack_bfloat16_vec_into_uint32_vec(bias_all);
    fixture->WriteBuffer(mesh_device, partials_dram, partials_packed);
    fixture->WriteBuffer(mesh_device, bias_dram, bias_packed);

    workload.add_program(device_range, std::move(program));
    fixture->RunProgram(mesh_device, workload);

    std::vector<uint32_t> result_packed;
    fixture->ReadBuffer(mesh_device, out_dram, result_packed);
    auto result_tilized_sb = unpack_uint32_vec_into_bfloat16_vec(result_packed);

    // Compute golden and compare per iteration.
    bool all_ok = true;
    for (uint32_t iter = 0; iter < cfg.num_invocations; iter++) {
        const auto& b = bias_per_push_1d[(cfg.pattern == BiasPattern::OneTimeFront) ? 0 : iter];
        const auto& p = partials_per_iter_rowmajor[iter];
        std::vector<bfloat16> golden(cfg.M * cfg.N);
        for (uint32_t i = 0; i < cfg.M; i++) {
            for (uint32_t j = 0; j < cfg.N; j++) {
                float v = static_cast<float>(p[i * cfg.N + j]) + static_cast<float>(b[j]);
                if (cfg.post_bias_relu) {
                    v = std::max(0.0f, v);
                }
                golden[i * cfg.N + j] = bfloat16(v);
            }
        }

        // Slice the result: iter's worth of tiles, reassemble row-major, untilize.
        uint32_t tile_elems = TILE_HEIGHT * TILE_WIDTH;
        std::vector<bfloat16> iter_sb(
            result_tilized_sb.begin() + iter * partials_tiles_per_iter * tile_elems,
            result_tilized_sb.begin() + (iter + 1) * partials_tiles_per_iter * tile_elems);
        auto iter_rm = subblock_order_to_row_major(iter_sb, Mt, Nt, cfg.out_subblock_h, cfg.out_subblock_w);
        auto iter_untiled = untilize_nfaces(iter_rm, cfg.M, cfg.N);

        float pcc = pcc_bfloat16(golden, iter_untiled);
        log_info(
            LogTest,
            "iter={} M={} N={} sb_h={} sb_w={} pattern={} post_relu={} — PCC = {:.6f}",
            iter,
            cfg.M,
            cfg.N,
            cfg.out_subblock_h,
            cfg.out_subblock_w,
            cfg.pattern == BiasPattern::OneTimeFront ? "one_time" : "per_iter",
            cfg.post_bias_relu,
            pcc);
        if (pcc < pcc_threshold) {
            all_ok = false;
        }
    }
    return all_ok;
}

}  // namespace test_add_bias_bcast_rows

using test_add_bias_bcast_rows::BiasConfig;
using test_add_bias_bcast_rows::BiasPattern;
using test_add_bias_bcast_rows::run_add_bias_bcast_rows_test;

// (a) One-time-front, no PostBiasFn
TEST_F(MeshDispatchFixture, TensixAddBiasBcastRowsOneTimeFrontSingleIter) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_add_bias_bcast_rows_test(
            this,
            device,
            {.M = 64,
             .N = 64,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .num_invocations = 1,
             .pattern = BiasPattern::OneTimeFront}));
    }
}

TEST_F(MeshDispatchFixture, TensixAddBiasBcastRowsOneTimeFrontMultiIter) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_add_bias_bcast_rows_test(
            this,
            device,
            {.M = 64,
             .N = 64,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .num_invocations = 3,
             .pattern = BiasPattern::OneTimeFront}));
    }
}

// (b) Per-iter push + pop
TEST_F(MeshDispatchFixture, TensixAddBiasBcastRowsPerIterMultiIter) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_add_bias_bcast_rows_test(
            this,
            device,
            {.M = 64,
             .N = 64,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .num_invocations = 3,
             .pattern = BiasPattern::PerIterPush}));
    }
}

// Multi-subblock coverage
TEST_F(MeshDispatchFixture, TensixAddBiasBcastRowsMultiSubblockN) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_add_bias_bcast_rows_test(
            this,
            device,
            {.M = 64,
             .N = 128,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .num_invocations = 1,
             .pattern = BiasPattern::OneTimeFront}));
    }
}

TEST_F(MeshDispatchFixture, TensixAddBiasBcastRowsMultiSubblockBoth) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_add_bias_bcast_rows_test(
            this,
            device,
            {.M = 128,
             .N = 128,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .num_invocations = 1,
             .pattern = BiasPattern::OneTimeFront}));
    }
}

// (c) PostBiasFn = relu, one-time-front
TEST_F(MeshDispatchFixture, TensixAddBiasBcastRowsPostBiasRelu) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_add_bias_bcast_rows_test(
            this,
            device,
            {.M = 64,
             .N = 64,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .num_invocations = 1,
             .pattern = BiasPattern::OneTimeFront,
             .post_bias_relu = true}));
    }
}

// (c) PostBiasFn = relu + per-iter push
TEST_F(MeshDispatchFixture, TensixAddBiasBcastRowsPostBiasReluPerIter) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_add_bias_bcast_rows_test(
            this,
            device,
            {.M = 64,
             .N = 64,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .num_invocations = 2,
             .pattern = BiasPattern::PerIterPush,
             .post_bias_relu = true}));
    }
}

}  // namespace tt::tt_metal
