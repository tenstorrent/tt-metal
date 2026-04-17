// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Isolated integration tests for the matmul_block helper
// (ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp).
//
// Each test pins one dimension of the template parameter pack
// (transpose, packer_l1_acc, pack_last_to_interm, pack_relu,
// row_major_output, PostComputeFn) against a CPU golden. Helper signatures
// are verified regardless of the caller kernel that uses them in production.
//
// Pattern adapted from the dropped tests at base commit 23d9cbd4.
// Key adaptations for the current helper API:
//   - matmul_block CB arguments are runtime, not template parameters
//   - row_major_output is a new template bool (absolute-offset packing)
//   - PostComputeFn is a template type instead of a macro
//
// Uses reader_matmul_blocked.cpp + writer_unary.cpp. The reader walks DRAM
// blocks in K-block order; the writer streams c_16 tiles to DRAM in the
// order the helper packed them. For row_major_output=true the stream is
// row-major over the Mt×Nt grid; for row_major_output=false it is subblock
// major. The host readback reorders to row-major tile order before untilize.

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

// Host golden for the matmul. `transpose_b_tiles` mirrors the helper's
// template-level WH-transpose of B: each 32x32 tile of B is transposed
// before the multiply. The resulting effect for a matrix partitioned into
// 32x32 tiles is equivalent to transposing within each tile, which is what
// the helper's `transpose` flag does via the LLK `matmul_block(transpose=1)`.
static void golden_matmul(
    const std::vector<bfloat16>& a,
    const std::vector<bfloat16>& b,
    std::vector<bfloat16>& output,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    bool transpose_b_tiles = false,
    bool apply_relu = false) {
    auto get_b = [&](uint32_t k, uint32_t n) -> float {
        if (!transpose_b_tiles) {
            return static_cast<float>(b[k * N + n]);
        }
        // Within the tile that contains (k, n), swap row/col.
        uint32_t tile_r = k / TILE_HEIGHT;
        uint32_t tile_c = n / TILE_WIDTH;
        uint32_t inner_r = k % TILE_HEIGHT;
        uint32_t inner_c = n % TILE_WIDTH;
        // Transposed index within the tile: swap inner_r and inner_c.
        uint32_t src_k = tile_r * TILE_HEIGHT + inner_c;
        uint32_t src_n = tile_c * TILE_WIDTH + inner_r;
        return static_cast<float>(b[src_k * N + src_n]);
    };

    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; k++) {
                acc += static_cast<float>(a[(i * K) + k]) * get_b(k, j);
            }
            if (apply_relu) {
                acc = std::max(0.0f, acc);
            }
            output[(i * N) + j] = bfloat16(acc);
        }
    }
}

// Rearrange tilized A from row-major tile order to block-column (K-block) order
// to match reader_matmul_blocked's layout.
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
    uint32_t M;               // rows (elements)
    uint32_t N;               // cols (elements)
    uint32_t K;               // inner dim (elements)
    uint32_t out_subblock_h;  // output sub-block height in tiles
    uint32_t out_subblock_w;  // output sub-block width in tiles
    uint32_t in0_block_w;     // K-dimension block size in tiles
    uint32_t batch = 1;

    // Helper feature flags
    bool transpose = false;          // B tiles WH-transpose
    bool packer_l1_acc = false;      // HW L1 accumulation across K-blocks
    bool pack_relu = false;          // helper applies relu on the last pack
    bool row_major_output = false;   // absolute-offset packing
    bool post_compute_relu = false;  // PostComputeFn = ReluPostCompute
    // NOTE: pack_last_to_interm is out-of-scope here because writer_unary only
    // drains c_16. It is exercised by the production fused-bias kernel plus
    // the add_bias_bcast_rows isolated test.
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
    uint32_t num_output_tiles = Mt * Nt * cfg.batch;

    uint32_t single_tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;
    tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    CoreCoord core({0, 0});

    distributed::MeshWorkload workload;
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    Program program{};

    uint32_t dram_size_a = single_tile_size * Mt * Kt * cfg.batch;
    uint32_t dram_size_b = single_tile_size * Kt * Nt * cfg.batch;
    uint32_t dram_size_c = single_tile_size * Mt * Nt * cfg.batch;

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

    uint32_t in0_cb_tiles = in0_block_num_tiles * 2;
    CircularBufferConfig cb_in0_config =
        CircularBufferConfig(in0_cb_tiles * single_tile_size, {{CBIndex::c_0, cb_data_format}})
            .set_page_size(CBIndex::c_0, single_tile_size);
    CreateCircularBuffer(program, core, cb_in0_config);

    uint32_t in1_cb_tiles = in1_block_num_tiles * 2;
    CircularBufferConfig cb_in1_config =
        CircularBufferConfig(in1_cb_tiles * single_tile_size, {{CBIndex::c_1, cb_data_format}})
            .set_page_size(CBIndex::c_1, single_tile_size);
    CreateCircularBuffer(program, core, cb_in1_config);

    // out (c_16) and interm (c_24) share the same L1 address space — matches
    // legacy multicast factory layout. The helper's shared-memory-protection
    // reserve covers the sequential (non-row-major) path on non-last K-blocks.
    uint32_t num_out_cb_tiles = Mt * Nt;
    std::map<uint8_t, tt::DataFormat> partials_and_out_spec = {
        {CBIndex::c_24, cb_data_format}, {CBIndex::c_16, cb_data_format}};
    CircularBufferConfig cb_out_config =
        CircularBufferConfig(num_out_cb_tiles * single_tile_size, partials_and_out_spec)
            .set_page_size(CBIndex::c_24, single_tile_size)
            .set_page_size(CBIndex::c_16, single_tile_size);
    CreateCircularBuffer(program, core, cb_out_config);

    auto reader_id = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_matmul_blocked.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    // Writer strategy:
    //   - All tests use writer_unary: streams tiles from c_16 to DRAM in CB
    //     order. For row_major_output tests that means row-major element
    //     layout; for subblock-order tests we reorder host-side (see below).
    //   - pack_last_to_interm is out of scope for the current writer: the
    //     helper packs to c_24 in that mode, and writer_unary only consumes
    //     c_16. The mode IS exercised indirectly via the production fused-bias
    //     kernel's real-matmul path; here we stick to pack_last_to_interm=false.
    auto writer_id = CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
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
        out_subblock_h,
        out_subblock_w,
        out_subblock_num_tiles,
        cfg.batch};

    std::map<std::string, std::string> defines;
    if (cfg.transpose) {
        defines["HELPER_TRANSPOSE"] = "1";
    }
    if (cfg.packer_l1_acc) {
        defines["HELPER_PACKER_L1_ACC"] = "1";
    }
    if (cfg.pack_relu) {
        defines["HELPER_PACK_RELU"] = "1";
    }
    if (cfg.row_major_output) {
        defines["HELPER_ROW_MAJOR_OUTPUT"] = "1";
    }
    if (cfg.post_compute_relu) {
        defines["HELPER_POST_COMPUTE_RELU"] = "1";
    }

    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/test_matmul_block_helper_compute.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_args, .defines = defines});

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
         num_blocks * cfg.batch,
         in0_block_num_tiles,
         in1_block_num_tiles,
         in0_block_size_bytes,
         in1_block_size_bytes});

    // writer_unary streams c_16 tiles straight to DRAM in CB order.
    SetRuntimeArgs(program, writer_id, core, {dst_dram->address(), 0, num_output_tiles});

    // Random inputs. For ReLU tests use a symmetric distribution so the
    // output has both signs (ReLU fires on meaningful data). For non-ReLU
    // tests stick to non-negative inputs so the PCC check is stable.
    std::mt19937 rng(42);
    const bool need_relu = cfg.pack_relu || cfg.post_compute_relu;
    std::uniform_real_distribution<float> dist(need_relu ? -1.0f : 0.0f, 1.0f);

    std::vector<bfloat16> src0_vec(cfg.M * cfg.K * cfg.batch);
    std::vector<bfloat16> src1_vec(cfg.K * cfg.N * cfg.batch);
    for (auto& v : src0_vec) {
        v = bfloat16(dist(rng));
    }
    for (auto& v : src1_vec) {
        v = bfloat16(dist(rng));
    }

    // Golden per batch slice
    const bool apply_relu = cfg.pack_relu || cfg.post_compute_relu;
    std::vector<bfloat16> golden_vec(cfg.M * cfg.N * cfg.batch, bfloat16(0.0f));
    for (uint32_t b = 0; b < cfg.batch; b++) {
        std::vector<bfloat16> slice_a(src0_vec.begin() + b * cfg.M * cfg.K, src0_vec.begin() + (b + 1) * cfg.M * cfg.K);
        std::vector<bfloat16> slice_b(src1_vec.begin() + b * cfg.K * cfg.N, src1_vec.begin() + (b + 1) * cfg.K * cfg.N);
        std::vector<bfloat16> slice_golden(cfg.M * cfg.N, bfloat16(0.0f));
        golden_matmul(slice_a, slice_b, slice_golden, cfg.M, cfg.N, cfg.K, cfg.transpose, apply_relu);
        std::copy(slice_golden.begin(), slice_golden.end(), golden_vec.begin() + b * cfg.M * cfg.N);
    }

    // Tilize per batch, reorder A to block-column
    std::vector<bfloat16> src0_all_tilized;
    std::vector<bfloat16> src1_all_tilized;
    src0_all_tilized.reserve(src0_vec.size());
    src1_all_tilized.reserve(src1_vec.size());
    for (uint32_t b = 0; b < cfg.batch; b++) {
        std::vector<bfloat16> slice_a(src0_vec.begin() + b * cfg.M * cfg.K, src0_vec.begin() + (b + 1) * cfg.M * cfg.K);
        std::vector<bfloat16> slice_b(src1_vec.begin() + b * cfg.K * cfg.N, src1_vec.begin() + (b + 1) * cfg.K * cfg.N);
        auto t_a = tilize_nfaces(slice_a, cfg.M, cfg.K);
        auto t_b = tilize_nfaces(slice_b, cfg.K, cfg.N);
        t_a = reorder_tiles_to_block_column(t_a, Mt, Kt, in0_block_w);
        src0_all_tilized.insert(src0_all_tilized.end(), t_a.begin(), t_a.end());
        src1_all_tilized.insert(src1_all_tilized.end(), t_b.begin(), t_b.end());
    }

    auto src0_packed = pack_bfloat16_vec_into_uint32_vec(src0_all_tilized);
    auto src1_packed = pack_bfloat16_vec_into_uint32_vec(src1_all_tilized);
    fixture->WriteBuffer(mesh_device, src0_dram, src0_packed);
    fixture->WriteBuffer(mesh_device, src1_dram, src1_packed);

    workload.add_program(device_range, std::move(program));
    fixture->RunProgram(mesh_device, workload);

    std::vector<uint32_t> result_packed;
    fixture->ReadBuffer(mesh_device, dst_dram, result_packed);
    auto result_tilized = unpack_uint32_vec_into_bfloat16_vec(result_packed);

    // Writer_unary streams tiles in CB order. Depending on the helper's pack
    // strategy, the CB holds tiles in:
    //   - row-major tile order over the Mt×Nt grid (row_major_output=true)
    //   - subblock-major order within row-groups (row_major_output=false)
    // Reassemble to row-major tile order for untilize_nfaces.
    const uint32_t tile_elems = TILE_HEIGHT * TILE_WIDTH;
    const uint32_t slice_tile_elems = Mt * Nt * tile_elems;
    std::vector<bfloat16> untilized(cfg.M * cfg.N * cfg.batch, bfloat16(0.0f));

    for (uint32_t b = 0; b < cfg.batch; b++) {
        std::vector<bfloat16> slice(
            result_tilized.begin() + b * slice_tile_elems, result_tilized.begin() + (b + 1) * slice_tile_elems);
        std::vector<bfloat16> slice_row_major;

        if (cfg.row_major_output) {
            slice_row_major = std::move(slice);
        } else {
            // Reassemble subblock-major tiles to row-major tile order.
            slice_row_major.resize(slice_tile_elems);
            uint32_t src = 0;
            const uint32_t num_sb_w = in1_num_subblocks;
            const uint32_t num_row_groups = in0_num_subblocks;
            for (uint32_t g = 0; g < num_row_groups; g++) {
                for (uint32_t sb = 0; sb < num_sb_w; sb++) {
                    for (uint32_t h = 0; h < out_subblock_h; h++) {
                        for (uint32_t w = 0; w < out_subblock_w; w++) {
                            uint32_t row_tile = g * out_subblock_h + h;
                            uint32_t col_tile = sb * out_subblock_w + w;
                            uint32_t dst_tile_idx = row_tile * Nt + col_tile;
                            std::copy(
                                slice.begin() + src,
                                slice.begin() + src + tile_elems,
                                slice_row_major.begin() + dst_tile_idx * tile_elems);
                            src += tile_elems;
                        }
                    }
                }
            }
        }

        auto u = untilize_nfaces(slice_row_major, cfg.M, cfg.N);
        std::copy(u.begin(), u.end(), untilized.begin() + b * cfg.M * cfg.N);
    }

    float pcc = pcc_bfloat16(golden_vec, untilized);
    log_info(
        LogTest,
        "M={} N={} K={} blk_w={} sub_h={} sub_w={} nb={} batch={} "
        "flags=[T={} L1={} R={} RM={} PCR={}] — PCC = {:.6f} (thresh {:.4f})",
        cfg.M,
        cfg.N,
        cfg.K,
        cfg.in0_block_w,
        cfg.out_subblock_h,
        cfg.out_subblock_w,
        num_blocks,
        cfg.batch,
        cfg.transpose,
        cfg.packer_l1_acc,
        cfg.pack_relu,
        cfg.row_major_output,
        cfg.post_compute_relu,
        pcc,
        pcc_threshold);

    return pcc > pcc_threshold;
}

}  // namespace test_matmul_block_helper

using test_matmul_block_helper::BlockMatmulConfig;
using test_matmul_block_helper::run_matmul_block_helper_test;

// ─── Baseline: all flags off ────────────────────────────────────────────────
TEST_F(MeshDispatchFixture, TensixMatmulBlockHelperBaseline) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_block_helper_test(
            this, device, {.M = 64, .N = 64, .K = 64, .out_subblock_h = 2, .out_subblock_w = 2, .in0_block_w = 2}));
    }
}

// ─── num_k_blocks variations (spill/reload path) ────────────────────────────
TEST_F(MeshDispatchFixture, TensixMatmulBlockHelperTwoKBlocks) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_block_helper_test(
            this, device, {.M = 64, .N = 64, .K = 128, .out_subblock_h = 2, .out_subblock_w = 2, .in0_block_w = 2}));
    }
}

TEST_F(MeshDispatchFixture, TensixMatmulBlockHelperFourKBlocks) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_block_helper_test(
            this, device, {.M = 64, .N = 64, .K = 256, .out_subblock_h = 2, .out_subblock_w = 2, .in0_block_w = 2}));
    }
}

// ─── Multi-subblock M / N / both (sharded-bmm bug regression guard) ─────────
// Pre-branch, sharded_matmul[bmm] corrupted data when in1 num_subblocks > 1
// because the legacy writer read row-major while the helper packed subblock
// order. Locked-in here by exercising in1_num_subblocks=2.
TEST_F(MeshDispatchFixture, TensixMatmulBlockHelperMultiSubblockN) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_block_helper_test(
            this, device, {.M = 64, .N = 128, .K = 64, .out_subblock_h = 2, .out_subblock_w = 2, .in0_block_w = 2}));
    }
}

TEST_F(MeshDispatchFixture, TensixMatmulBlockHelperMultiSubblockM) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_block_helper_test(
            this, device, {.M = 128, .N = 64, .K = 64, .out_subblock_h = 2, .out_subblock_w = 2, .in0_block_w = 2}));
    }
}

TEST_F(MeshDispatchFixture, TensixMatmulBlockHelperMultiSubblockBoth) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_block_helper_test(
            this, device, {.M = 128, .N = 128, .K = 64, .out_subblock_h = 2, .out_subblock_w = 2, .in0_block_w = 2}));
    }
}

// ─── Batch > 1 ──────────────────────────────────────────────────────────────
TEST_F(MeshDispatchFixture, TensixMatmulBlockHelperBatch2) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_block_helper_test(
            this,
            device,
            {.M = 64, .N = 64, .K = 64, .out_subblock_h = 2, .out_subblock_w = 2, .in0_block_w = 2, .batch = 2}));
    }
}

// ─── Feature flag coverage ──────────────────────────────────────────────────

TEST_F(MeshDispatchFixture, TensixMatmulBlockHelperPackerL1AccMultiBlock) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_block_helper_test(
            this,
            device,
            {.M = 64,
             .N = 64,
             .K = 128,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .in0_block_w = 2,
             .packer_l1_acc = true}));
    }
}

TEST_F(MeshDispatchFixture, TensixMatmulBlockHelperPackRelu) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_block_helper_test(
            this,
            device,
            {.M = 64,
             .N = 64,
             .K = 64,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .in0_block_w = 2,
             .pack_relu = true}));
    }
}

TEST_F(MeshDispatchFixture, TensixMatmulBlockHelperPackReluMultiBlock) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_block_helper_test(
            this,
            device,
            {.M = 64,
             .N = 64,
             .K = 128,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .in0_block_w = 2,
             .pack_relu = true}));
    }
}

TEST_F(MeshDispatchFixture, TensixMatmulBlockHelperRowMajorOutput) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_block_helper_test(
            this,
            device,
            {.M = 64,
             .N = 64,
             .K = 64,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .in0_block_w = 2,
             .row_major_output = true}));
    }
}

TEST_F(MeshDispatchFixture, TensixMatmulBlockHelperRowMajorOutputMultiSubblockN) {
    // Exercises row-major packing with N-subblocking: per-row-group reserve/push,
    // absolute-offset column positions stride across subblocks. The bmm corruption
    // regression test.
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_block_helper_test(
            this,
            device,
            {.M = 64,
             .N = 128,
             .K = 64,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .in0_block_w = 2,
             .row_major_output = true}));
    }
}

TEST_F(MeshDispatchFixture, TensixMatmulBlockHelperRowMajorOutputMultiBlock) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_block_helper_test(
            this,
            device,
            {.M = 64,
             .N = 64,
             .K = 128,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .in0_block_w = 2,
             .row_major_output = true}));
    }
}

TEST_F(MeshDispatchFixture, TensixMatmulBlockHelperPostComputeRelu) {
    // PostComputeFn: relu applied per sub-block on last K-block before packing.
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_block_helper_test(
            this,
            device,
            {.M = 64,
             .N = 64,
             .K = 64,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .in0_block_w = 2,
             .post_compute_relu = true}));
    }
}

TEST_F(MeshDispatchFixture, TensixMatmulBlockHelperPostComputeReluMultiBlock) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_block_helper_test(
            this,
            device,
            {.M = 64,
             .N = 64,
             .K = 128,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .in0_block_w = 2,
             .post_compute_relu = true}));
    }
}

// ─── Transpose (WH transpose of srcB tiles) ────────────────────────────────
// Toggles the `transpose` template bool on matmul_block<>. The LLK transposes
// each srcB tile WH on the fly. Golden mirrors by swapping row/col within
// every tile of B.
TEST_F(MeshDispatchFixture, TensixMatmulBlockHelperTransposeB) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_block_helper_test(
            this,
            device,
            {.M = 64,
             .N = 64,
             .K = 64,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .in0_block_w = 2,
             .transpose = true}));
    }
}

// ─── Combined-flag stress ───────────────────────────────────────────────────
TEST_F(MeshDispatchFixture, TensixMatmulBlockHelperL1AccRowMajorMultiSubblockN) {
    for (const auto& device : devices_) {
        ASSERT_TRUE(run_matmul_block_helper_test(
            this,
            device,
            {.M = 64,
             .N = 128,
             .K = 128,
             .out_subblock_h = 2,
             .out_subblock_w = 2,
             .in0_block_w = 2,
             .packer_l1_acc = true,
             .row_major_output = true}));
    }
}

}  // namespace tt::tt_metal
