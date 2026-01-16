// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common/command_queue_fixture.hpp"

#include <algorithm>
#include <functional>
#include <random>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/distributed.hpp>
#include "tt_metal/test_utils/deprecated/tensor.hpp"
#include <tt-metalium/tilize_utils.hpp>

using std::vector;
using namespace tt;
using std::string;
using namespace tt::tt_metal;

static std::vector<bfloat16> make_identity_scale_tile() {
    std::vector<bfloat16> tile(tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH, static_cast<bfloat16>(1.0f));
    return tile;
}

static std::vector<bfloat16> make_prev_max_matrix(
    uint32_t q_chunk_size, float min_val, float max_val, std::vector<bfloat16>& prev_max_first_col_out) {
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(min_val, max_val);

    const uint32_t rows = q_chunk_size * 32;
    const uint32_t cols = 32;
    std::vector<bfloat16> mat(rows * cols);
    prev_max_first_col_out.resize(rows);

    for (uint32_t r = 0; r < rows; ++r) {
        float v = dist(rng);
        prev_max_first_col_out[r] = static_cast<bfloat16>(v);
        for (uint32_t c = 0; c < cols; ++c) {
            mat[(r * cols) + c] = static_cast<bfloat16>(v);
        }
    }
    return mat;
}

static std::vector<bfloat16> golden_reduce_c(
    const std::vector<bfloat16>& qk_im_rm,
    const std::vector<bfloat16>& prev_max_first_col,
    uint32_t q_chunk_size,
    uint32_t k_chunk_size,
    bool do_eltwise_max) {
    const uint32_t rows = q_chunk_size * 32;
    const uint32_t cols = k_chunk_size * 32;
    const uint32_t stats_cols = 32;

    bool prev_max_larger = false;

    // Produce tiles: rows tiles, each 32x32. We'll set first column to the row max (replicated down the column).
    std::vector<bfloat16> out(rows * stats_cols, static_cast<bfloat16>(0.0f));

    for (uint32_t r = 0; r < rows; ++r) {
        float row_max = -std::numeric_limits<float>::infinity();
        for (uint32_t c = 0; c < cols; ++c) {
            row_max = std::max(row_max, static_cast<float>(qk_im_rm[(r * cols) + c]));
        }
        if (do_eltwise_max) {
            float working_row_max = row_max;
            row_max = std::max(row_max, static_cast<float>(prev_max_first_col[r]));
            if (row_max > working_row_max) {
                prev_max_larger = true;
            }
        }
        out[(r * stats_cols) + 0] = static_cast<bfloat16>(row_max);
    }

    // Expand to tiles (each row has a 32x32 tile). Fill only first column; others left as zero and not checked.
    std::vector<bfloat16> tilized(rows * stats_cols * 32, static_cast<bfloat16>(0.0f));
    // However, for comparison we will only use untilized first column from device output; no need to mirror tile layout
    // here.
    log_info(LogTest, "prev_max_larger: {}", prev_max_larger);
    return out;  // row-major (rows x 32), first column holds value
}

static float compare_first_col_mse(
    const std::vector<bfloat16>& result_rm, const std::vector<bfloat16>& golden_first_col_rm) {
    // result_rm is rows x 32 row-major (from untilize); we compare only column 0 vs golden_first_col_rm (rows x 32,
    // only col0 set)
    const uint32_t rows = golden_first_col_rm.size() / 32;
    float mse = 0.0f;
    for (uint32_t r = 0; r < rows; ++r) {
        float a = static_cast<float>(result_rm[(r * 32) + 0]);
        float b = static_cast<float>(golden_first_col_rm[(r * 32) + 0]);
        float d = a - b;
        mse += d * d;
    }
    mse /= rows;
    return mse;
}

static bool test_sdpa_reduce_c(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& mesh_device,
    uint32_t q_chunk_size,
    uint32_t k_chunk_size,
    bool fp32_dest_acc_en,
    bool do_eltwise_max,
    const std::string& kernel_path,
    const std::string& kernel_name) {
    bool pass = true;

    log_info(
        LogTest,
        "Running {} test with q_chunk_size: {}, k_chunk_size: {}, fp32_dest_acc_en: {}, "
        "do_eltwise_max: {}",
        kernel_name,
        q_chunk_size,
        k_chunk_size,
        fp32_dest_acc_en,
        do_eltwise_max);

    // Get device and command queue from mesh
    tt_metal::IDevice* device = mesh_device->get_devices().at(0);
    tt_metal::distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue(0);

    tt_metal::Program program = tt_metal::CreateProgram();

    CoreCoord core = {0, 0};

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

    // Create sharded buffers for CB inputs/outputs
    auto qk_im_buffer = CreateBuffer(qk_im_buffer_config);
    auto prev_max_buffer = CreateBuffer(stats_buffer_config);
    auto out_max_buffer = CreateBuffer(stats_buffer_config);
    auto identity_scale_buffer = CreateBuffer(one_tile_buffer_config);

    // Create CBs and point them to sharded buffers
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
        tt::tt_metal::CircularBufferConfig(stats_num_tiles * cb_tile_size, {{cb_out_max_id, cb_df}})
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
    // For this testing, use granularity of 1
    compute_defines["REDUCE_GRANULARITY"] = "1";
    compute_defines["LOG2_REDUCE_GRANULARITY"] = "0";

    tt_metal::CreateKernel(
        program,
        kernel_path,
        core,
        tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
            .defines = compute_defines});

    // Prepare inputs
    SHAPE qk_im_shape = {1, 1, q_chunk_size * 32, k_chunk_size * 32};
    tt::deprecated::Tensor<bfloat16> qk_im_tensor = tt::deprecated::initialize_tensor<bfloat16>(
        qk_im_shape, tt::deprecated::Initialize::RANDOM, -50, 50, 0 /* seed */);

    vector<uint32_t> qk_im;
    auto qk_im_tilized = tilize_nfaces(qk_im_tensor.get_values(), q_chunk_size * 32, k_chunk_size * 32);
    qk_im = pack_bfloat16_vec_into_uint32_vec(qk_im_tilized);
    tt_metal::detail::WriteToBuffer(qk_im_buffer, qk_im);

    std::vector<bfloat16> prev_max_first_col;
    auto prev_max_rm = make_prev_max_matrix(q_chunk_size, 25.0f, 65.0f, prev_max_first_col);
    auto prev_max_tilized = tilize_nfaces(prev_max_rm, q_chunk_size * 32, 32);
    auto prev_max_uint_vec = pack_bfloat16_vec_into_uint32_vec(prev_max_tilized);
    tt_metal::detail::WriteToBuffer(prev_max_buffer, prev_max_uint_vec);

    auto identity_scale_tile = make_identity_scale_tile();
    auto identity_scale_uint_vec = pack_bfloat16_vec_into_uint32_vec(identity_scale_tile);
    tt_metal::detail::WriteToBuffer(identity_scale_buffer, identity_scale_uint_vec);

    // Execute program using MeshWorkload
    tt_metal::distributed::MeshWorkload workload;
    tt_metal::distributed::MeshCoordinate zero_coord =
        tt_metal::distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    tt_metal::distributed::MeshCoordinateRange device_range =
        tt_metal::distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, std::move(program));
    tt_metal::distributed::EnqueueMeshWorkload(cq, workload, false);
    tt_metal::distributed::Finish(cq);

    // Read outputs
    std::vector<uint32_t> out_max_vec;
    tt_metal::detail::ReadFromBuffer(out_max_buffer, out_max_vec);
    auto out_max_bfp16 = unpack_uint32_vec_into_bfloat16_vec(out_max_vec);
    auto out_max_rm = untilize_nfaces(out_max_bfp16, q_chunk_size * 32, 32);

    // Golden
    auto golden_first_col_rm =
        golden_reduce_c(qk_im_tensor.get_values(), prev_max_first_col, q_chunk_size, k_chunk_size, do_eltwise_max);

    float mse = compare_first_col_mse(out_max_rm, golden_first_col_rm);
    const float max_mse = 0.0f;  // expect exact max match in first column
    log_info(LogTest, "{} first-col mse: {} (do_eltwise: {})", kernel_name, mse, do_eltwise_max);
    if (mse > max_mse) {
        log_error(LogTest, "{} first-col mse: {} > {} (do_eltwise: {})", kernel_name, mse, max_mse, do_eltwise_max);
        pass = false;
    }

    return pass;
}

// NIGHTLY_ prefix ensures this test only runs in nightly CI pipelines
TEST_F(UnitMeshCQSingleCardFixture, NIGHTLY_SdpaReduceC) {
    bool pass = true;

    /**
     * Parameters to sweep over for correctness.
     */
    std::vector<uint32_t> q_chunk_sizes = {1, 2, 4, 8};
    std::vector<uint32_t> k_chunk_sizes = {1, 2, 4, 8, 16};
    std::vector<bool> fp32_dest_acc_ens = {false, true};
    std::vector<bool> do_eltwise = {false, true};

    /**
     * These parameters are the same as the SDPA sprint-2 perfomance test parameters.
     * Uncomment to measure perf of the test we care most about.
     */
    // std::vector<uint32_t> q_chunk_sizes = {8};
    // std::vector<uint32_t> k_chunk_sizes = {16};
    // std::vector<uint32_t> head_dims = {128};
    // std::vector<bool> fp32_dest_acc_ens = {false};
    // std::vector<bool> do_eltwise = {false, true};

    // Test both implementations
    std::vector<std::pair<std::string, std::string>> kernel_variants = {
        {"tests/tt_metal/tt_metal/test_kernels/misc/sdpa/reduce_c/compute.cpp", "reduce_c"},
        {"tests/tt_metal/tt_metal/test_kernels/misc/sdpa/reduce_block_max_row/compute.cpp", "reduce_block_max_row"}};

    for (const auto& [kernel_path, kernel_name] : kernel_variants) {
        log_info(LogTest, "Testing kernel variant: {}", kernel_name);
        for (uint32_t q_chunk_size : q_chunk_sizes) {
            for (uint32_t k_chunk_size : k_chunk_sizes) {
                for (bool fp32_dest_acc_en : fp32_dest_acc_ens) {
                    for (bool do_elt : do_eltwise) {
                        bool this_passed = test_sdpa_reduce_c(
                            devices_[0],
                            q_chunk_size,
                            k_chunk_size,
                            fp32_dest_acc_en,
                            do_elt,
                            kernel_path,
                            kernel_name);
                        if (!this_passed) {
                            log_error(
                                LogTest,
                                "Test Failed for kernel: {}, q_chunk_size: {}, k_chunk_size: {}"
                                "fp32_dest_acc_en: "
                                "{}, do_eltwise: {}",
                                kernel_name,
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
    }

    ASSERT_TRUE(pass);
}
