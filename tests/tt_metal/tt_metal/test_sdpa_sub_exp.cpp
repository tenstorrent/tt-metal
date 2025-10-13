// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using std::vector;
using namespace tt;
using std::string;

void print_faces(std::vector<bfloat16> data, string name) {
    std::cout << name << ": " << std::endl;
    // int index = 0;

    int tile_index = 0;
    int face_index = 0;
    for (int i = 0; i < data.size(); i++) {
        if (i % 256 == 0) {
            std::cout << "Tile " << tile_index / 4 << std::endl;
            std::cout << "Face = " << face_index << std::endl;
            face_index++;
            tile_index++;
            if (face_index == 4) {
                face_index = 0;
            }
        }
        std::cout << static_cast<float>(data.at(i)) << ", ";
        if ((i + 1) % 16 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

std::pair<std::vector<bfloat16>, std::vector<bfloat16>> golden(
    const std::vector<bfloat16>& qk_im,
    const std::vector<bfloat16>& cur_max,
    const std::vector<bfloat16>& cur_sum,
    const uint32_t q_chunk_size,
    uint32_t k_chunk_size,
    uint32_t head_dim) {
    std::vector<bfloat16> sub_exp_result(q_chunk_size * k_chunk_size * 32 * 32, static_cast<bfloat16>(0.0f));
    std::vector<bfloat16> cur_sum_result(q_chunk_size * 32 * 32, static_cast<bfloat16>(0.0f));

    float scale = 1.0 / std::sqrt(head_dim);

    uint32_t num_qk_cols = k_chunk_size * 32;
    uint32_t num_stats_cols = 32;

    for (int row = 0; row < q_chunk_size * 32; row++) {
        float cur_max_val = static_cast<float>(cur_max[row * num_stats_cols]);
        for (int col = 0; col < k_chunk_size * 32; col++) {
            float qk = static_cast<float>(qk_im[row * num_qk_cols + col]);
            float sub_exp = std::exp((qk - cur_max_val) * scale);
            sub_exp_result[row * num_qk_cols + col] = static_cast<bfloat16>(sub_exp);
            uint32_t sum_tile_col = col % 32;
            cur_sum_result[row * num_stats_cols + sum_tile_col] += sub_exp;
        }
    }

    return std::make_pair(sub_exp_result, cur_sum_result);
}

float compare_mse(const std::vector<bfloat16>& result, const std::vector<bfloat16>& golden) {
    float mse = 0;
    for (int i = 0; i < result.size(); i++) {
        mse += std::pow(static_cast<float>(result[i]) - static_cast<float>(golden[i]), 2);
    }
    mse /= result.size();
    return mse;
}

bool test_sdpa_sub_exp(
    tt_metal::IDevice* device, uint32_t q_chunk_size, uint32_t k_chunk_size, uint32_t head_dim, bool fp32_dest_acc_en) {
    bool pass = true;

    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    log_info(
        LogTest,
        "Running sdpa_sub_exp test with q_chunk_size: {}, k_chunk_size: {}, head_dim: {}, fp32_dest_acc_en: {}",
        q_chunk_size,
        k_chunk_size,
        head_dim,
        fp32_dest_acc_en);

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
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
        auto cb_qk_im_config =
            tt::tt_metal::CircularBufferConfig(qk_im_num_tiles * cb_tile_size, {{cb_qk_im_id, cb_df}})
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

        float scale = 1.0 / std::sqrt(head_dim);
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
            "tests/tt_metal/tt_metal/test_kernels/misc/sdpa/sub_exp/compute.cpp",
            core,
            tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi2,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .math_approx_mode = false,
                .compile_args = compute_kernel_args,
                .defines = compute_defines});

        ////////////////////////////////////////////////////////////////////////////
        //                      Compile Application
        ////////////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        SHAPE qk_im_shape = {1, 1, q_chunk_size * 32, k_chunk_size * 32};
        tt::deprecated::Tensor<bfloat16> qk_im_tensor = tt::deprecated::initialize_tensor<bfloat16>(
            qk_im_shape, tt::deprecated::Initialize::RANDOM, -50, 50, 0 /* seed */);

        vector<uint32_t> qk_im;
        auto qk_im_tilized = tilize_nfaces(qk_im_tensor.get_values(), q_chunk_size * 32, k_chunk_size * 32);
        qk_im = pack_bfloat16_vec_into_uint32_vec(qk_im_tilized);
        tt_metal::detail::WriteToBuffer(qk_im_buffer, qk_im);

        // Cur max buffer initialized with 50.0f in the first column
        std::vector<bfloat16> cur_max_values(q_chunk_size * 32 * 32, static_cast<bfloat16>(0.0f));
        for (int i = 0; i < q_chunk_size * 32 * 32; i += 32) {
            cur_max_values[i] = static_cast<bfloat16>(50.0f);
        }

        auto cur_max_values_tilized = tilize_nfaces(cur_max_values, q_chunk_size * 32, 32);
        auto cur_max_uint_vec = pack_bfloat16_vec_into_uint32_vec(cur_max_values_tilized);
        tt_metal::detail::WriteToBuffer(cur_max_buffer, cur_max_uint_vec);

        // Initialized to zero
        std::vector<bfloat16> cur_sum_values(q_chunk_size * 32 * 32, static_cast<bfloat16>(0.0f));
        auto cur_sum_values_tilized = tilize_nfaces(cur_sum_values, q_chunk_size * 32, 32);
        auto cur_sum_uint_vec = pack_bfloat16_vec_into_uint32_vec(cur_sum_values_tilized);
        tt_metal::detail::WriteToBuffer(cur_sum_buffer, cur_sum_uint_vec);

        // print_faces(qk_im_tilized, "qk_im_tilized");
        // print_faces(cur_max_values_tilized, "cur_max_values_tilized");
        // print_faces(cur_sum_values_tilized, "cur_sum_values_tilized");

        tt_metal::detail::LaunchProgram(device, program, true);
        std::vector<uint32_t> qk_im_result_vec;
        tt_metal::detail::ReadFromBuffer(qk_im_buffer, qk_im_result_vec);
        auto qk_im_result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(qk_im_result_vec);
        auto qk_im_result_rm = untilize_nfaces(qk_im_result_bfp16, q_chunk_size * 32, k_chunk_size * 32);
        // print_faces(qk_im_result_bfp16, "qk_im_result");

        std::vector<uint32_t> cur_sum_result_vec;
        tt_metal::detail::ReadFromBuffer(cur_sum_buffer, cur_sum_result_vec);
        auto cur_sum_result_bfp16 = unpack_uint32_vec_into_bfloat16_vec(cur_sum_result_vec);
        auto cur_sum_result_rm = untilize_nfaces(cur_sum_result_bfp16, q_chunk_size * 32, 32);
        // print_faces(cur_sum_result_bfp16, "cur_sum_result");

        auto [sub_exp_golden_result, cur_sum_golden_result] =
            golden(qk_im_tensor.get_values(), cur_max_values, cur_sum_values, q_chunk_size, k_chunk_size, head_dim);

        float qk_im_mse = compare_mse(qk_im_result_rm, sub_exp_golden_result);
        float cur_sum_mse = compare_mse(cur_sum_result_rm, cur_sum_golden_result);
        const float sub_exp_max_mse = 1e-4;

        log_info(LogTest, "qk_im_mse: {}, cur_sum_mse: {}", qk_im_mse, cur_sum_mse);
        if (qk_im_mse > sub_exp_max_mse) {
            log_error(LogTest, "qk_im_mse: {} > max_mse: {}", qk_im_mse, sub_exp_max_mse);
            pass = false;
        }
        const float cur_sum_max_mse = 1.5e-3;
        if (cur_sum_mse > cur_sum_max_mse) {
            log_error(LogTest, "cur_sum_mse: {} > max_mse: {}", cur_sum_mse, cur_sum_max_mse);
            pass = false;
        }

    } catch (const std::exception& e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    return pass;
}

int main(int argc, char** argv) {
    bool pass = true;
    // Once this test is uplifted to use fast dispatch, this can be removed.
    char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
    putenv(env);
    ////////////////////////////////////////////////////////////////////////////
    //                      Initial Runtime Args Parse
    ////////////////////////////////////////////////////////////////////////////
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

    for (uint32_t q_chunk_size : q_chunk_sizes) {
        for (uint32_t k_chunk_size : k_chunk_sizes) {
            for (uint32_t head_dim : head_dims) {
                for (bool fp32_dest_acc_en : fp32_dest_acc_ens) {
                    bool this_passed =
                        test_sdpa_sub_exp(device, q_chunk_size, k_chunk_size, head_dim, fp32_dest_acc_en);
                    if (!this_passed) {
                        log_error(
                            LogTest,
                            "Test Failed for q_chunk_size: {}, k_chunk_size: {}, head_dim: {}, fp32_dest_acc_en: {}",
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

    pass &= tt_metal::CloseDevice(device);

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass, "Error");
}
