// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/bfloat16.hpp>
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include <tt-metalium/tilize_utils.hpp>
#include <sstream>
#include <iostream>

using std::vector;
using namespace tt;
using std::string;

static void transpose_tiles_inplace_row_major(std::vector<bfloat16>& rm, uint32_t rows, uint32_t cols) {
    const uint32_t tile_h = 32;
    const uint32_t tile_w = 32;
    uint32_t tiles_h = rows / tile_h;
    uint32_t tiles_w = cols / tile_w;
    std::vector<bfloat16> copy = rm;
    for (uint32_t th = 0; th < tiles_h; ++th) {
        for (uint32_t tw = 0; tw < tiles_w; ++tw) {
            for (uint32_t i = 0; i < tile_h; ++i) {
                for (uint32_t j = 0; j < tile_w; ++j) {
                    uint32_t src_r = th * tile_h + i;
                    uint32_t src_c = tw * tile_w + j;
                    uint32_t dst_r = th * tile_h + j;
                    uint32_t dst_c = tw * tile_w + i;
                    rm[dst_r * cols + dst_c] = copy[src_r * cols + src_c];
                }
            }
        }
    }
}

// static std::vector<bfloat16> golden_reduce_c_transposed(
//     const std::vector<bfloat16>& qk_im_rm_transposed,
//     const std::vector<bfloat16>& prev_max_first_col,
//     uint32_t q_chunk_size,
//     uint32_t k_chunk_size,
//     bool do_eltwise_max) {
//     const uint32_t rows = q_chunk_size * 32;
//     const uint32_t cols = k_chunk_size * 32;
//     const uint32_t stats_cols = 32;

//     // Undo tile transpose to compute row-wise max on original orientation
//     std::vector<bfloat16> qk_im_rm = qk_im_rm_transposed;
//     transpose_tiles_inplace_row_major(qk_im_rm, rows, cols);

//     std::vector<bfloat16> out(rows * stats_cols, static_cast<bfloat16>(0.0f));
//     for (uint32_t r = 0; r < rows; ++r) {
//         float row_max = -std::numeric_limits<float>::infinity();
//         for (uint32_t c = 0; c < cols; ++c) {
//             row_max = std::max(row_max, static_cast<float>(qk_im_rm[r * cols + c]));
//         }
//         if (do_eltwise_max) {
//             row_max = std::max(row_max, static_cast<float>(prev_max_first_col[r]));
//         }
//         out[r * stats_cols + 0] = static_cast<bfloat16>(row_max);
//     }

//     // Re-apply tile transpose for the stats output (rows x 32)
//     transpose_tiles_inplace_row_major(out, rows, 32);
//     return out;
// }

// static float compare_first_col_mse(const std::vector<bfloat16>& result_rm, const std::vector<bfloat16>& golden_rm) {
//     // Operates on row-major data, where each tile in the row-major data has been transposed.
//     const uint32_t n_tiles = result_rm.size() / (32 * 32);
//     const uint32_t rows = golden_rm.size() / 32;
//     float mse = 0.0f;
//     uint32_t start_idx = 0;
//     for (uint32_t t = 0; t < n_tiles; ++t) {
//         for (uint32_t r = 0; r < 32; ++r) {
//             float a = static_cast<float>(result_rm[start_idx + r]);
//             float b = static_cast<float>(golden_rm[start_idx + r]);
//             float d = a - b;
//             mse += d * d;
//         }
//         start_idx += 32 * 32;
//     }
//     mse /= rows;
//     return mse;
// }

static bool test_sdpa_reduce_c_transposed(
    tt_metal::IDevice* device,
    uint32_t q_chunk_size,
    uint32_t k_chunk_size,
    bool fp32_dest_acc_en,
    bool do_eltwise_max) {
    bool pass = true;

    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    log_info(
        LogTest,
        "Running sdpa_reduce_c_transposed test with q_chunk_size: {}, k_chunk_size: {}, "
        "fp32_dest_acc_en: {}, do_eltwise_max: {}",
        q_chunk_size,
        k_chunk_size,
        fp32_dest_acc_en,
        do_eltwise_max);

    tt_metal::Program program = tt_metal::CreateProgram();
    CoreCoord core = {0, 0};

    auto cb_df = tt::DataFormat::Float16_b;
    auto cb_tile_size = tt::tile_size(cb_df);

    uint32_t qk_im_num_tiles = q_chunk_size * k_chunk_size;
    uint32_t stats_num_tiles = q_chunk_size;     // rows
    uint32_t out_cols_num_tiles = k_chunk_size;  // cols

    auto qk_im_buffer_config = tt::tt_metal::ShardedBufferConfig{
        .device = device,
        .size = qk_im_num_tiles * cb_tile_size,
        .page_size = cb_tile_size,
        .buffer_type = tt::tt_metal::BufferType::L1,
        .buffer_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = tt::tt_metal::ShardSpecBuffer(
            CoreRangeSet(std::set<CoreRange>({CoreRange(core, core)})),
            {q_chunk_size * tt::constants::TILE_HEIGHT, k_chunk_size * tt::constants::TILE_WIDTH},
            tt::tt_metal::ShardOrientation::ROW_MAJOR,
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            {q_chunk_size, k_chunk_size})};

    auto stats_buffer_config = tt::tt_metal::ShardedBufferConfig{
        .device = device,
        .size = stats_num_tiles * cb_tile_size,
        .page_size = cb_tile_size,
        .buffer_type = tt::tt_metal::BufferType::L1,
        .buffer_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = tt::tt_metal::ShardSpecBuffer(
            CoreRangeSet(std::set<CoreRange>({CoreRange(core, core)})),
            {stats_num_tiles * tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            tt::tt_metal::ShardOrientation::ROW_MAJOR,
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            {stats_num_tiles, 1})};

    // Output buffer sized by cols (k_chunk_size)
    auto cols_buffer_config = tt::tt_metal::ShardedBufferConfig{
        .device = device,
        .size = out_cols_num_tiles * cb_tile_size,
        .page_size = cb_tile_size,
        .buffer_type = tt::tt_metal::BufferType::L1,
        .buffer_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = tt::tt_metal::ShardSpecBuffer(
            CoreRangeSet(std::set<CoreRange>({CoreRange(core, core)})),
            {out_cols_num_tiles * tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            tt::tt_metal::ShardOrientation::ROW_MAJOR,
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            {out_cols_num_tiles, 1})};

    auto one_tile_buffer_config = tt::tt_metal::ShardedBufferConfig{
        .device = device,
        .size = cb_tile_size,
        .page_size = cb_tile_size,
        .buffer_type = tt::tt_metal::BufferType::L1,
        .buffer_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = tt::tt_metal::ShardSpecBuffer(
            CoreRangeSet(std::set<CoreRange>({CoreRange(core, core)})),
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            tt::tt_metal::ShardOrientation::ROW_MAJOR,
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            {1, 1})};

    auto qk_im_buffer = CreateBuffer(qk_im_buffer_config);
    auto prev_max_buffer = CreateBuffer(stats_buffer_config);
    auto out_max_buffer = CreateBuffer(cols_buffer_config);
    auto identity_scale_buffer = CreateBuffer(one_tile_buffer_config);

    auto cb_qk_im_id = tt::CBIndex::c_0;
    auto cb_qk_im_config = tt::tt_metal::CircularBufferConfig(qk_im_num_tiles * cb_tile_size, {{cb_qk_im_id, cb_df}})
                               .set_page_size(cb_qk_im_id, cb_tile_size)
                               .set_globally_allocated_address(*qk_im_buffer);
    tt_metal::CreateCircularBuffer(program, core, cb_qk_im_config);

    auto cb_prev_max_id = tt::CBIndex::c_1;
    auto cb_prev_max_config =
        tt::tt_metal::CircularBufferConfig(stats_num_tiles * cb_tile_size, {{cb_prev_max_id, cb_df}})
            .set_page_size(cb_prev_max_id, cb_tile_size)
            .set_globally_allocated_address(*prev_max_buffer);
    tt_metal::CreateCircularBuffer(program, core, cb_prev_max_config);

    auto cb_out_max_id = tt::CBIndex::c_2;
    auto cb_out_max_config =
        tt::tt_metal::CircularBufferConfig(out_cols_num_tiles * cb_tile_size, {{cb_out_max_id, cb_df}})
            .set_page_size(cb_out_max_id, cb_tile_size)
            .set_globally_allocated_address(*out_max_buffer);
    tt_metal::CreateCircularBuffer(program, core, cb_out_max_config);

    auto cb_identity_scale_id = tt::CBIndex::c_3;
    auto cb_identity_scale_config =
        tt::tt_metal::CircularBufferConfig(1 * cb_tile_size, {{cb_identity_scale_id, cb_df}})
            .set_page_size(cb_identity_scale_id, cb_tile_size)
            .set_globally_allocated_address(*identity_scale_buffer);
    tt_metal::CreateCircularBuffer(program, core, cb_identity_scale_config);

    std::vector<uint32_t> compute_kernel_args = {
        cb_qk_im_id,
        cb_prev_max_id,
        cb_out_max_id,
        cb_identity_scale_id,
        q_chunk_size,
        k_chunk_size,
        static_cast<uint32_t>(do_eltwise_max ? 1 : 0)};
    std::map<std::string, std::string> compute_defines;
    // unnecessary defines
    compute_defines["SUB_EXP_GRANULARITY"] = "0";
    compute_defines["LOG2_SUB_EXP_GRANULARITY"] = "0";
    compute_defines["STATS_GRANULARITY"] = "0";
    compute_defines["LOG2_STATS_GRANULARITY"] = "0";
    compute_defines["MUL_BCAST_GRANULARITY"] = "0";
    compute_defines["LOG2_MUL_BCAST_GRANULARITY"] = "0";
    compute_defines["DHT_GRANULARITY"] = "0";
    compute_defines["LOG2_DHT_GRANULARITY"] = "0";
    compute_defines["EXP_APPROX_MODE"] = "0";

    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/sdpa/reduce_c_transposed/compute.cpp",
        core,
        tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
            .defines = compute_defines});

    // Inputs: tile-transposed qk_im and prev_max
    SHAPE qk_im_shape = {1, 1, q_chunk_size * 32, k_chunk_size * 32};
    tt::deprecated::Tensor<bfloat16> qk_im_tensor = tt::deprecated::initialize_tensor<bfloat16>(
        qk_im_shape, tt::deprecated::Initialize::RANDOM, -50, 50, 0 /* seed */);

    std::vector<bfloat16> qk_rm = qk_im_tensor.get_values();
    transpose_tiles_inplace_row_major(qk_rm, q_chunk_size * 32, k_chunk_size * 32);
    auto qk_im_tilized = tilize_nfaces(qk_rm, q_chunk_size * 32, k_chunk_size * 32);
    auto qk_im_uint = pack_bfloat16_vec_into_uint32_vec(qk_im_tilized);
    tt_metal::detail::WriteToBuffer(qk_im_buffer, qk_im_uint);

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(25.0f, 65.0f);
    const uint32_t rows = q_chunk_size * 32;
    std::vector<bfloat16> prev_max_first_col(rows);
    std::vector<bfloat16> prev_max_rm(rows * 32, static_cast<bfloat16>(0.0f));
    for (uint32_t r = 0; r < rows; ++r) {
        float v = dist(rng);
        prev_max_first_col[r] = static_cast<bfloat16>(v);
        prev_max_rm[r * 32 + 0] = static_cast<bfloat16>(v);
    }
    // Transpose tiles of the stats matrix (rows x 32)
    transpose_tiles_inplace_row_major(prev_max_rm, rows, 32);
    auto prev_max_tilized = tilize_nfaces(prev_max_rm, rows, 32);
    auto prev_max_uint = pack_bfloat16_vec_into_uint32_vec(prev_max_tilized);
    tt_metal::detail::WriteToBuffer(prev_max_buffer, prev_max_uint);

    // identity scale tile of ones
    std::vector<bfloat16> scale_tile(
        tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH, static_cast<bfloat16>(1.0f));
    auto scale_uint = pack_bfloat16_vec_into_uint32_vec(scale_tile);
    tt_metal::detail::WriteToBuffer(identity_scale_buffer, scale_uint);

    tt_metal::detail::LaunchProgram(device, program, true);

    std::vector<uint32_t> out_max_vec;
    tt_metal::detail::ReadFromBuffer(out_max_buffer, out_max_vec);
    auto out_max_bfp16 = unpack_uint32_vec_into_bfloat16_vec(out_max_vec);
    // Kernel outputs 1 x k_chunk_size tiles; untilize to 32 x (k_chunk_size*32)
    auto out_max_rm = untilize_nfaces(out_max_bfp16, 32, k_chunk_size * 32);

    // New MSE: compare OUT row0 per column against per-column maxima across all rows (1xcols semantics)
    float mse = 0.0f;
    {
        uint32_t rows_total = q_chunk_size * 32;
        uint32_t width_total = k_chunk_size * 32;
        uint32_t out_width = width_total;
        for (uint32_t col = 0; col < width_total; ++col) {
            float gmax = -std::numeric_limits<float>::infinity();
            for (uint32_t r = 0; r < rows_total; ++r) {
                float v = static_cast<float>(qk_rm[r * width_total + col]);
                if (v > gmax) {
                    gmax = v;
                }
            }
            float out_v = static_cast<float>(out_max_rm[0 * out_width + col]);
            float d = out_v - gmax;
            mse += d * d;
        }
        mse /= static_cast<float>(k_chunk_size * 32);
    }

    // Debug: print first row of OUT and GOLDEN per tile in paired order
    {
        uint32_t out_width = k_chunk_size * 32;
        // GOLDEN per-column maxima for row0 per tile (matches 1xcols reduction semantics)
        uint32_t rows_total = q_chunk_size * 32;
        uint32_t width_total = k_chunk_size * 32;

        auto print_golden_row_for_tile = [&](uint32_t tile_col_idx) {
            std::ostringstream gs;
            gs << "GOLDEN row0 tile" << tile_col_idx << ": ";
            for (uint32_t c = 0; c < 32; ++c) {
                float gmax = -std::numeric_limits<float>::infinity();
                uint32_t col_index = tile_col_idx * 32 + c;
                for (uint32_t jr = 0; jr < rows_total; ++jr) {
                    float v = static_cast<float>(qk_rm[jr * width_total + col_index]);
                    if (v > gmax) {
                        gmax = v;
                    }
                }
                gs << gmax;
                if (c != 31) {
                    gs << ' ';
                }
            }
            std::cout << gs.str() << std::endl;
        };

        // OUT tile 0, row 0 then GOLDEN tile 0
        {
            std::ostringstream oss0;
            oss0 << "OUT row0 tile0:    ";
            for (uint32_t c = 0; c < 32; ++c) {
                float v = static_cast<float>(out_max_rm[0 * out_width + 0 + c]);
                oss0 << v;
                if (c != 31) {
                    oss0 << ' ';
                }
            }
            std::cout << oss0.str() << std::endl;
            print_golden_row_for_tile(0);
        }

        // OUT tile 1, row 0 then GOLDEN tile 1 (if exists)
        if (k_chunk_size >= 2) {
            std::ostringstream oss1;
            oss1 << "OUT row0 tile1:    ";
            for (uint32_t c = 0; c < 32; ++c) {
                float v = static_cast<float>(out_max_rm[0 * out_width + 32 + c]);
                oss1 << v;
                if (c != 31) {
                    oss1 << ' ';
                }
            }
            std::cout << oss1.str() << std::endl;
            print_golden_row_for_tile(1);
        }
    }

    if (mse > 0.0f) {
        log_error(LogTest, "reduce_c_transposed mse: {} > 0", mse);
        pass = false;
    }

    return pass;
}

int main(int argc, char** argv) {
    bool pass = true;
    char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
    putenv(env);

    int device_id = 0;
    tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);

    /**
     * Parameters to sweep over for correctness.
     */
    std::vector<uint32_t> q_chunk_sizes = {1, 2, 4, 8};  // rows
    std::vector<uint32_t> k_chunk_sizes = {1, 2, 4, 8};  // cols
    std::vector<bool> fp32_dest_acc_ens = {false};  //, true};
    std::vector<bool> do_eltwise = {false};         //, true};
    /**
     * These parameters are the same as the SDPA sprint-2 perfomance test parameters.
     * Uncomment to measure perf of the test we care most about.
     */
    // std::vector<uint32_t> q_chunk_sizes = {8};
    // std::vector<uint32_t> k_chunk_sizes = {16};
    // std::vector<bool> fp32_dest_acc_ens = {false};
    // std::vector<bool> do_eltwise = {false, true};

    for (uint32_t q_chunk_size : q_chunk_sizes) {
        for (uint32_t k_chunk_size : k_chunk_sizes) {
            if (q_chunk_size * k_chunk_size > 8) {
                log_info(
                    LogTest, "Skipping test: q_chunk_size * k_chunk_size > 8 ({}, {})", q_chunk_size, k_chunk_size);
                continue;
            }
            for (bool fp32_dest_acc_en : fp32_dest_acc_ens) {
                for (bool do_elt : do_eltwise) {
                    bool this_passed =
                        test_sdpa_reduce_c_transposed(device, q_chunk_size, k_chunk_size, fp32_dest_acc_en, do_elt);
                    if (!this_passed) {
                        log_error(
                            LogTest,
                            "Test Failed for q_chunk_size: {}, k_chunk_size: {}, fp32_dest_acc_en: {}, do_eltwise: {}",
                            q_chunk_size,
                            k_chunk_size,
                            fp32_dest_acc_en,
                            do_elt);
                    }
                    pass &= this_passed;
                }
            }
        }
    }

    pass &= tt_metal::CloseDevice(device);

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass, "Error");
}
