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

using std::vector;
using namespace tt;
using std::string;

static void transpose_tiles_inplace_row_major(std::vector<bfloat16>& rm, uint32_t rows, uint32_t cols) {
    // Transpose each 32x32 tile independently in row-major tensor of shape rows x cols (both multiples of 32)
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

static std::pair<std::vector<bfloat16>, std::vector<bfloat16>> golden_rows_transposed(
    const std::vector<bfloat16>& qk_im_rm_transposed_tiles,
    const std::vector<bfloat16>& cur_max_rm_transposed_tiles,
    uint32_t q_chunk_size,
    uint32_t k_chunk_size,
    uint32_t head_dim) {
    const uint32_t rows = q_chunk_size * 32;
    const uint32_t cols = k_chunk_size * 32;

    // Undo tile-level transpose to compute golden as if original (non-transposed) tiles
    std::vector<bfloat16> qk_im_rm = qk_im_rm_transposed_tiles;
    std::vector<bfloat16> cur_max_rm = cur_max_rm_transposed_tiles;
    transpose_tiles_inplace_row_major(qk_im_rm, rows, cols);
    transpose_tiles_inplace_row_major(cur_max_rm, rows, 32);

    // Compute same golden as sub_exp: exp((qk - cur_max) * scale) and partial row sums
    std::vector<bfloat16> sub_exp_result(rows * cols, static_cast<bfloat16>(0.0f));
    std::vector<bfloat16> cur_sum_result(rows * 32, static_cast<bfloat16>(0.0f));

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (uint32_t r = 0; r < rows; r++) {
        float cur_max_val = static_cast<float>(cur_max_rm[r * 32 + 0]);
        for (uint32_t c = 0; c < cols; c++) {
            float qk = static_cast<float>(qk_im_rm[r * cols + c]);
            float sub_exp = std::exp((qk - cur_max_val) * scale);
            sub_exp_result[r * cols + c] = static_cast<bfloat16>(sub_exp);
            uint32_t sum_tile_col = c % 32;
            cur_sum_result[r * 32 + sum_tile_col] += sub_exp;
        }
    }

    // Now re-apply tile-level transpose to match device output layout
    transpose_tiles_inplace_row_major(sub_exp_result, rows, cols);
    transpose_tiles_inplace_row_major(cur_sum_result, rows, 32);

    return std::make_pair(sub_exp_result, cur_sum_result);
}

static float compare_mse(const std::vector<bfloat16>& result, const std::vector<bfloat16>& golden) {
    float mse = 0.0f;
    for (size_t i = 0; i < result.size(); i++) {
        float d = static_cast<float>(result[i]) - static_cast<float>(golden[i]);
        mse += d * d;
    }
    mse /= static_cast<float>(result.size());
    return mse;
}

static bool test_sdpa_sub_exp_rows_transposed(
    tt_metal::IDevice* device, uint32_t q_chunk_size, uint32_t k_chunk_size, uint32_t head_dim, bool fp32_dest_acc_en) {
    bool pass = true;

    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    log_info(
        LogTest,
        "Running sdpa_sub_exp_rows_transposed test with q_chunk_size: {}, k_chunk_size: {}, head_dim: {}, "
        "fp32_dest_acc_en: {}",
        q_chunk_size,
        k_chunk_size,
        head_dim,
        fp32_dest_acc_en);

    tt_metal::Program program = tt_metal::CreateProgram();

    CoreCoord core = {0, 0};

    uint32_t dst_tiles = fp32_dest_acc_en ? 4 : 8;
    uint32_t sub_exp_granularity = std::min(k_chunk_size, dst_tiles);
    uint32_t log2_sub_exp_granularity = std::log2(sub_exp_granularity);
    TT_FATAL(
        sub_exp_granularity == (1 << log2_sub_exp_granularity),
        "sub_exp_granularity must be a power of 2. Got {}.",
        sub_exp_granularity);

    auto cb_df = tt::DataFormat::Float16_b;
    auto cb_tile_size = tt::tile_size(cb_df);

    uint32_t qk_im_num_tiles = q_chunk_size * k_chunk_size;
    uint32_t stats_num_tiles = q_chunk_size;

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

    // Create sharded buffers for CB inputs
    auto qk_im_buffer = CreateBuffer(qk_im_buffer_config);
    auto cur_max_buffer = CreateBuffer(stats_buffer_config);
    auto cur_sum_buffer = CreateBuffer(stats_buffer_config);

    // Create CBs and point them to sharded buffers
    auto cb_qk_im_id = tt::CBIndex::c_0;
    auto cb_qk_im_config = tt::tt_metal::CircularBufferConfig(qk_im_num_tiles * cb_tile_size, {{cb_qk_im_id, cb_df}})
                               .set_page_size(cb_qk_im_id, cb_tile_size)
                               .set_globally_allocated_address(*qk_im_buffer);
    tt_metal::CreateCircularBuffer(program, core, cb_qk_im_config);

    auto cb_cur_max_id = tt::CBIndex::c_1;
    auto cb_cur_max_config =
        tt::tt_metal::CircularBufferConfig(stats_num_tiles * cb_tile_size, {{cb_cur_max_id, cb_df}})
            .set_page_size(cb_cur_max_id, cb_tile_size)
            .set_globally_allocated_address(*cur_max_buffer);
    tt_metal::CreateCircularBuffer(program, core, cb_cur_max_config);

    auto cb_cur_sum_id = tt::CBIndex::c_2;
    auto cb_cur_sum_config =
        tt::tt_metal::CircularBufferConfig(stats_num_tiles * cb_tile_size, {{cb_cur_sum_id, cb_df}})
            .set_page_size(cb_cur_sum_id, cb_tile_size)
            .set_globally_allocated_address(*cur_sum_buffer);
    tt_metal::CreateCircularBuffer(program, core, cb_cur_sum_config);

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    union {
        float f;
        uint32_t u;
    } scale_union;
    scale_union.f = scale;

    std::vector<uint32_t> compute_kernel_args = {
        cb_qk_im_id, cb_cur_max_id, cb_cur_sum_id, q_chunk_size, k_chunk_size, scale_union.u};
    std::map<std::string, std::string> compute_defines;
    compute_defines["SUB_EXP_GRANULARITY"] = std::to_string(sub_exp_granularity);
    compute_defines["LOG2_SUB_EXP_GRANULARITY"] = std::to_string(log2_sub_exp_granularity);
    // unnecessary defines
    compute_defines["STATS_GRANULARITY"] = "0";
    compute_defines["LOG2_STATS_GRANULARITY"] = "0";
    compute_defines["MUL_BCAST_GRANULARITY"] = "0";
    compute_defines["LOG2_MUL_BCAST_GRANULARITY"] = "0";
    compute_defines["DHT_GRANULARITY"] = "0";
    compute_defines["LOG2_DHT_GRANULARITY"] = "0";
    compute_defines["EXP_APPROX_MODE"] = "0";

    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/sdpa/sub_exp_rows_transposed/compute.cpp",
        core,
        tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
            .defines = compute_defines});

    // Execute: prepare inputs transposed at tile-level
    SHAPE qk_im_shape = {1, 1, q_chunk_size * 32, k_chunk_size * 32};
    tt::deprecated::Tensor<bfloat16> qk_im_tensor = tt::deprecated::initialize_tensor<bfloat16>(
        qk_im_shape, tt::deprecated::Initialize::RANDOM, -50, 50, 0 /* seed */);

    std::vector<bfloat16> qk_rm = qk_im_tensor.get_values();
    transpose_tiles_inplace_row_major(qk_rm, q_chunk_size * 32, k_chunk_size * 32);
    auto qk_im_tilized = tilize_nfaces(qk_rm, q_chunk_size * 32, k_chunk_size * 32);
    auto qk_im_uint = pack_bfloat16_vec_into_uint32_vec(qk_im_tilized);
    tt_metal::detail::WriteToBuffer(qk_im_buffer, qk_im_uint);

    // cur_max: single column per tile row; transpose within tiles still keeps first column as first row
    std::vector<bfloat16> cur_max_values(q_chunk_size * 32 * 32, static_cast<bfloat16>(0.0f));
    for (int i = 0; i < static_cast<int>(q_chunk_size * 32 * 32); i += 32) {
        cur_max_values[i] = static_cast<bfloat16>(50.0f);
    }
    transpose_tiles_inplace_row_major(cur_max_values, q_chunk_size * 32, 32);
    auto cur_max_tilized = tilize_nfaces(cur_max_values, q_chunk_size * 32, 32);
    auto cur_max_uint = pack_bfloat16_vec_into_uint32_vec(cur_max_tilized);
    tt_metal::detail::WriteToBuffer(cur_max_buffer, cur_max_uint);

    std::vector<bfloat16> cur_sum_values(q_chunk_size * 32 * 32, static_cast<bfloat16>(0.0f));
    auto cur_sum_tilized = tilize_nfaces(cur_sum_values, q_chunk_size * 32, 32);
    auto cur_sum_uint = pack_bfloat16_vec_into_uint32_vec(cur_sum_tilized);
    tt_metal::detail::WriteToBuffer(cur_sum_buffer, cur_sum_uint);

    tt_metal::detail::LaunchProgram(device, program, true);

    std::vector<uint32_t> qk_im_result_vec;
    tt_metal::detail::ReadFromBuffer(qk_im_buffer, qk_im_result_vec);
    auto qk_im_result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(qk_im_result_vec);
    auto qk_im_result_rm = untilize_nfaces(qk_im_result_bfp16, q_chunk_size * 32, k_chunk_size * 32);

    std::vector<uint32_t> cur_sum_result_vec;
    tt_metal::detail::ReadFromBuffer(cur_sum_buffer, cur_sum_result_vec);
    auto cur_sum_result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(cur_sum_result_vec);
    auto cur_sum_result_rm = untilize_nfaces(cur_sum_result_bfp16, q_chunk_size * 32, 32);

    auto [sub_exp_golden_result, cur_sum_golden_result] =
        golden_rows_transposed(qk_rm, cur_max_values, q_chunk_size, k_chunk_size, head_dim);

    float qk_im_mse = compare_mse(qk_im_result_rm, sub_exp_golden_result);
    float cur_sum_mse = compare_mse(cur_sum_result_rm, cur_sum_golden_result);
    const float sub_exp_max_mse = 1e-4f;
    const float cur_sum_max_mse = 1.5e-3f;

    log_info(LogTest, "qk_im_mse_t: {}, cur_sum_mse_t: {}", qk_im_mse, cur_sum_mse);
    if (qk_im_mse > sub_exp_max_mse) {
        log_error(LogTest, "qk_im_mse_t: {} > max_mse: {}", qk_im_mse, sub_exp_max_mse);
        pass = false;
    }
    if (cur_sum_mse > cur_sum_max_mse) {
        log_error(LogTest, "cur_sum_mse_t: {} > max_mse: {}", cur_sum_mse, cur_sum_max_mse);
        pass = false;
    }

    return pass;
}

int main(int argc, char** argv) {
    bool pass = true;
    char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
    putenv(env);

    std::vector<std::string> input_args(argv, argv + argc);

    int device_id = 0;
    tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);

    /**
     * Parameters to sweep over for correctness.
     */
    std::vector<uint32_t> q_chunk_sizes = {1, 2, 4, 8};
    std::vector<uint32_t> k_chunk_sizes = {1, 2, 4, 8, 16};
    std::vector<uint32_t> head_dims = {64, 128, 256};
    std::vector<bool> fp32_dest_acc_ens = {false, true};

    /**
     * These parameters are the same as the SDPA sprint-2 perfomance test parameters.
     * Uncomment to measure perf of the test we care most about.
     */
    // std::vector<uint32_t> q_chunk_sizes = {8};
    // std::vector<uint32_t> k_chunk_sizes = {16};
    // std::vector<uint32_t> head_dims = {128};
    // std::vector<bool> fp32_dest_acc_ens = {false};

    try {
        for (uint32_t q_chunk_size : q_chunk_sizes) {
            for (uint32_t k_chunk_size : k_chunk_sizes) {
                for (uint32_t head_dim : head_dims) {
                    for (bool fp32_dest_acc_en : fp32_dest_acc_ens) {
                        bool this_passed = test_sdpa_sub_exp_rows_transposed(
                            device, q_chunk_size, k_chunk_size, head_dim, fp32_dest_acc_en);
                        if (!this_passed) {
                            log_error(
                                LogTest,
                                "Test Failed for q_chunk_size: {}, k_chunk_size: {}, head_dim: {}, fp32_dest_acc_en: "
                                "{}",
                                q_chunk_size,
                                k_chunk_size,
                                head_dim,
                                fp32_dest_acc_en);
                        }
                        pass &= this_passed;
                    }
                }
            }
        }

    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }
    pass &= tt_metal::CloseDevice(device);

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass, "Error");
}
