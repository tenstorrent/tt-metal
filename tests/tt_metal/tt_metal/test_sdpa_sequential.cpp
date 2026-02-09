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

// Golden computation for Q @ K^T
static std::vector<bfloat16> golden_qk_matmul(
    const std::vector<bfloat16>& q_rm,
    const std::vector<bfloat16>& k_rm,
    uint32_t M,    // rows of Q
    uint32_t N,    // rows of K (cols after transpose)
    uint32_t K) {  // inner dimension

    std::vector<bfloat16> result(M * N);

    // Q @ K^T: (M x K) @ (K x N) = (M x N)
    for (uint32_t m = 0; m < M; ++m) {
        for (uint32_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                float q_val = static_cast<float>(q_rm[m * K + k]);
                float kt_val = static_cast<float>(k_rm[n * K + k]);  // K^T
                sum += q_val * kt_val;
            }
            result[m * N + n] = static_cast<bfloat16>(sum);
        }
    }

    return result;
}

static float compute_mse(const std::vector<bfloat16>& result, const std::vector<bfloat16>& golden) {
    if (result.size() != golden.size()) {
        log_error(LogTest, "Size mismatch: result={}, golden={}", result.size(), golden.size());
        return std::numeric_limits<float>::infinity();
    }

    float mse = 0.0f;
    for (size_t i = 0; i < result.size(); ++i) {
        float diff = static_cast<float>(result[i]) - static_cast<float>(golden[i]);
        mse += diff * diff;
    }
    mse /= result.size();
    return mse;
}

// Helper to create identity scale tile
static std::vector<bfloat16> make_identity_scale_tile() {
    std::vector<bfloat16> tile(tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH, static_cast<bfloat16>(1.0f));
    return tile;
}

static bool test_sdpa_sequential_phase1(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& mesh_device,
    uint32_t q_chunk_tiles,
    uint32_t k_chunk_tiles,
    uint32_t inner_dim_tiles,
    bool fp32_dest_acc_en) {
    bool pass = true;

    log_info(
        LogTest,
        "Running SDPA Sequential (QK Matmul only for debug) test with q_chunk_tiles: {}, k_chunk_tiles: {}, "
        "inner_dim_tiles: {}, fp32_dest_acc_en: {}",
        q_chunk_tiles,
        k_chunk_tiles,
        inner_dim_tiles,
        fp32_dest_acc_en);

    // Get device and command queue from mesh
    tt_metal::IDevice* device = mesh_device->get_devices().at(0);
    tt_metal::distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue(0);

    tt_metal::Program program = tt_metal::CreateProgram();

    CoreCoord core = {0, 0};

    auto cb_df = tt::DataFormat::Float16_b;
    auto cb_tile_size = tt::tile_size(cb_df);

    // Calculate buffer sizes
    uint32_t q_num_tiles = q_chunk_tiles * inner_dim_tiles;
    uint32_t k_num_tiles = k_chunk_tiles * inner_dim_tiles;
    uint32_t v_num_tiles = k_chunk_tiles * inner_dim_tiles;  // V has same dims as K
    uint32_t qk_num_tiles = q_chunk_tiles * k_chunk_tiles;
    uint32_t out_num_tiles = q_chunk_tiles * k_chunk_tiles;  // Output is QK after sub_exp

    // Create sharded buffer configs
    auto q_buffer_config = tt::tt_metal::ShardedBufferConfig{
        .device = device,
        .size = q_num_tiles * cb_tile_size,
        .page_size = cb_tile_size,
        .buffer_type = tt::tt_metal::BufferType::L1,
        .buffer_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = tt::tt_metal::ShardSpecBuffer(
            CoreRangeSet(std::set<CoreRange>({CoreRange(core, core)})),
            {q_chunk_tiles * tt::constants::TILE_HEIGHT, inner_dim_tiles * tt::constants::TILE_WIDTH},
            tt::tt_metal::ShardOrientation::ROW_MAJOR,
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            {q_chunk_tiles, inner_dim_tiles})};

    auto k_buffer_config = tt::tt_metal::ShardedBufferConfig{
        .device = device,
        .size = k_num_tiles * cb_tile_size,
        .page_size = cb_tile_size,
        .buffer_type = tt::tt_metal::BufferType::L1,
        .buffer_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = tt::tt_metal::ShardSpecBuffer(
            CoreRangeSet(std::set<CoreRange>({CoreRange(core, core)})),
            {k_chunk_tiles * tt::constants::TILE_HEIGHT, inner_dim_tiles * tt::constants::TILE_WIDTH},
            tt::tt_metal::ShardOrientation::ROW_MAJOR,
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            {k_chunk_tiles, inner_dim_tiles})};

    auto v_buffer_config = tt::tt_metal::ShardedBufferConfig{
        .device = device,
        .size = v_num_tiles * cb_tile_size,
        .page_size = cb_tile_size,
        .buffer_type = tt::tt_metal::BufferType::L1,
        .buffer_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = tt::tt_metal::ShardSpecBuffer(
            CoreRangeSet(std::set<CoreRange>({CoreRange(core, core)})),
            {k_chunk_tiles * tt::constants::TILE_HEIGHT, inner_dim_tiles * tt::constants::TILE_WIDTH},
            tt::tt_metal::ShardOrientation::ROW_MAJOR,
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            {k_chunk_tiles, inner_dim_tiles})};

    auto qk_buffer_config = tt::tt_metal::ShardedBufferConfig{
        .device = device,
        .size = qk_num_tiles * cb_tile_size,
        .page_size = cb_tile_size,
        .buffer_type = tt::tt_metal::BufferType::L1,
        .buffer_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = tt::tt_metal::ShardSpecBuffer(
            CoreRangeSet(std::set<CoreRange>({CoreRange(core, core)})),
            {q_chunk_tiles * tt::constants::TILE_HEIGHT, k_chunk_tiles * tt::constants::TILE_WIDTH},
            tt::tt_metal::ShardOrientation::ROW_MAJOR,
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            {q_chunk_tiles, k_chunk_tiles})};

    // Max buffer config (one tile per row for reduce output)
    auto max_buffer_config = tt::tt_metal::ShardedBufferConfig{
        .device = device,
        .size = q_chunk_tiles * cb_tile_size,
        .page_size = cb_tile_size,
        .buffer_type = tt::tt_metal::BufferType::L1,
        .buffer_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = tt::tt_metal::ShardSpecBuffer(
            CoreRangeSet(std::set<CoreRange>({CoreRange(core, core)})),
            {q_chunk_tiles * tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            tt::tt_metal::ShardOrientation::ROW_MAJOR,
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            {q_chunk_tiles, 1})};

    // Scale buffer config (single identity tile)
    auto scale_buffer_config = tt::tt_metal::ShardedBufferConfig{
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

    // Output buffer config (stores QK after sub_exp)
    auto out_buffer_config = tt::tt_metal::ShardedBufferConfig{
        .device = device,
        .size = out_num_tiles * cb_tile_size,
        .page_size = cb_tile_size,
        .buffer_type = tt::tt_metal::BufferType::L1,
        .buffer_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = tt::tt_metal::ShardSpecBuffer(
            CoreRangeSet(std::set<CoreRange>({CoreRange(core, core)})),
            {q_chunk_tiles * tt::constants::TILE_HEIGHT, k_chunk_tiles * tt::constants::TILE_WIDTH},
            tt::tt_metal::ShardOrientation::ROW_MAJOR,
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            {q_chunk_tiles, k_chunk_tiles})};

    // Create sharded buffers
    auto q_buffer = CreateBuffer(q_buffer_config);
    auto k_buffer = CreateBuffer(k_buffer_config);
    auto v_buffer = CreateBuffer(v_buffer_config);
    auto qk_buffer = CreateBuffer(qk_buffer_config);
    auto max_buffer = CreateBuffer(max_buffer_config);
    auto scale_buffer = CreateBuffer(scale_buffer_config);
    auto out_buffer = CreateBuffer(out_buffer_config);

    // Create circular buffers
    auto cb_q_id = tt::CBIndex::c_0;
    auto cb_q_config = tt::tt_metal::CircularBufferConfig(q_num_tiles * cb_tile_size, {{cb_q_id, cb_df}})
                           .set_page_size(cb_q_id, cb_tile_size)
                           .set_globally_allocated_address(*q_buffer);
    tt_metal::CreateCircularBuffer(program, core, cb_q_config);

    auto cb_k_id = tt::CBIndex::c_1;
    auto cb_k_config = tt::tt_metal::CircularBufferConfig(k_num_tiles * cb_tile_size, {{cb_k_id, cb_df}})
                           .set_page_size(cb_k_id, cb_tile_size)
                           .set_globally_allocated_address(*k_buffer);
    tt_metal::CreateCircularBuffer(program, core, cb_k_config);

    auto cb_v_id = tt::CBIndex::c_2;
    auto cb_v_config = tt::tt_metal::CircularBufferConfig(v_num_tiles * cb_tile_size, {{cb_v_id, cb_df}})
                           .set_page_size(cb_v_id, cb_tile_size)
                           .set_globally_allocated_address(*v_buffer);
    tt_metal::CreateCircularBuffer(program, core, cb_v_config);

    auto cb_qk_id = tt::CBIndex::c_3;
    auto cb_qk_config = tt::tt_metal::CircularBufferConfig(qk_num_tiles * cb_tile_size, {{cb_qk_id, cb_df}})
                            .set_page_size(cb_qk_id, cb_tile_size)
                            .set_globally_allocated_address(*qk_buffer);
    tt_metal::CreateCircularBuffer(program, core, cb_qk_config);

    auto cb_max_id = tt::CBIndex::c_4;
    auto cb_max_config = tt::tt_metal::CircularBufferConfig(q_chunk_tiles * cb_tile_size, {{cb_max_id, cb_df}})
                             .set_page_size(cb_max_id, cb_tile_size)
                             .set_globally_allocated_address(*max_buffer);
    tt_metal::CreateCircularBuffer(program, core, cb_max_config);

    auto cb_scale_id = tt::CBIndex::c_5;
    auto cb_scale_config = tt::tt_metal::CircularBufferConfig(cb_tile_size, {{cb_scale_id, cb_df}})
                               .set_page_size(cb_scale_id, cb_tile_size)
                               .set_globally_allocated_address(*scale_buffer);
    tt_metal::CreateCircularBuffer(program, core, cb_scale_config);

    auto cb_out_id = tt::CBIndex::c_6;
    auto cb_out_config = tt::tt_metal::CircularBufferConfig(out_num_tiles * cb_tile_size, {{cb_out_id, cb_df}})
                             .set_page_size(cb_out_id, cb_tile_size)
                             .set_globally_allocated_address(*out_buffer);
    tt_metal::CreateCircularBuffer(program, core, cb_out_config);

    // Compile-time arguments for compute kernel
    std::vector<uint32_t> compute_kernel_args = {
        cb_q_id,
        cb_k_id,
        cb_v_id,
        cb_qk_id,
        cb_max_id,
        cb_scale_id,
        cb_out_id,
        q_chunk_tiles,
        k_chunk_tiles,
        inner_dim_tiles};

    // Create compute kernel
    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/sdpa/sequential/compute.cpp",
        core,
        tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
            .defines = {}});

    // Prepare inputs
    uint32_t M = q_chunk_tiles * 32;
    uint32_t N = k_chunk_tiles * 32;
    uint32_t K = inner_dim_tiles * 32;

    SHAPE q_shape = {1, 1, M, K};
    SHAPE k_shape = {1, 1, N, K};
    SHAPE v_shape = {1, 1, N, K};  // V has same shape as K

    tt::deprecated::Tensor<bfloat16> q_tensor =
        tt::deprecated::initialize_tensor<bfloat16>(q_shape, tt::deprecated::Initialize::RANDOM, -2, 2, 100 /* seed */);

    tt::deprecated::Tensor<bfloat16> k_tensor =
        tt::deprecated::initialize_tensor<bfloat16>(k_shape, tt::deprecated::Initialize::RANDOM, -2, 2, 200 /* seed */);

    tt::deprecated::Tensor<bfloat16> v_tensor =
        tt::deprecated::initialize_tensor<bfloat16>(v_shape, tt::deprecated::Initialize::RANDOM, -2, 2, 300 /* seed */);

    // Tilize and write to buffers
    auto q_tilized = tilize_nfaces(q_tensor.get_values(), M, K);
    auto q_packed = pack_bfloat16_vec_into_uint32_vec(q_tilized);
    tt_metal::detail::WriteToBuffer(q_buffer, q_packed);

    auto k_tilized = tilize_nfaces(k_tensor.get_values(), N, K);
    auto k_packed = pack_bfloat16_vec_into_uint32_vec(k_tilized);
    tt_metal::detail::WriteToBuffer(k_buffer, k_packed);

    auto v_tilized = tilize_nfaces(v_tensor.get_values(), N, K);
    auto v_packed = pack_bfloat16_vec_into_uint32_vec(v_tilized);
    tt_metal::detail::WriteToBuffer(v_buffer, v_packed);

    // Create identity scale tile
    auto scale_tile = make_identity_scale_tile();
    auto scale_tilized = tilize_nfaces(scale_tile, 32, 32);
    auto scale_packed = pack_bfloat16_vec_into_uint32_vec(scale_tilized);
    tt_metal::detail::WriteToBuffer(scale_buffer, scale_packed);

    // Execute program
    tt_metal::distributed::MeshWorkload workload;
    tt_metal::distributed::MeshCoordinate zero_coord =
        tt_metal::distributed::MeshCoordinate::zero_coordinate(mesh_device->shape().dims());
    tt_metal::distributed::MeshCoordinateRange device_range =
        tt_metal::distributed::MeshCoordinateRange(zero_coord, zero_coord);
    workload.add_program(device_range, std::move(program));
    tt_metal::distributed::EnqueueMeshWorkload(cq, workload, false);
    tt_metal::distributed::Finish(cq);

    // Read final output
    std::vector<uint32_t> out_vec;
    tt_metal::detail::ReadFromBuffer(out_buffer, out_vec);
    auto out_bfp16 = unpack_uint32_vec_into_bfloat16_vec(out_vec);
    auto out_rm = untilize_nfaces(out_bfp16, M, K);

    // Compute golden for QK matmul only (for debugging)
    auto golden_out = golden_qk_matmul(q_tensor.get_values(), k_tensor.get_values(), M, N, K);

    // Compare result
    float mse = compute_mse(out_rm, golden_out);
    const float max_mse = 0.1f;
    log_info(LogTest, "QK Matmul MSE: {}", mse);

    if (mse > max_mse) {
        log_error(LogTest, "QK Matmul MSE: {} > {}", mse, max_mse);
        pass = false;
    }

    return pass;
}

// NIGHTLY_ prefix ensures this test only runs in nightly CI pipelines
TEST_F(UnitMeshCQSingleCardFixture, NIGHTLY_SdpaSequentialFullSequence) {
    bool pass = true;

    // Full sequence: QK→Reduce→SubExp→QK→Reduce→SubExp→QKV
    // Q: 2 tiles (rows) x 4 tiles (inner dim)
    // K: 4 tiles (rows) x 4 tiles (inner dim)
    // V: 4 tiles (rows) x 4 tiles (inner dim)
    // QK: 2 tiles (rows) x 4 tiles (cols)
    // Output: 2 tiles (rows) x 4 tiles (inner dim)
    uint32_t q_chunk_tiles = 2;
    uint32_t k_chunk_tiles = 4;
    uint32_t inner_dim_tiles = 4;  // head_dim / 32 (e.g., 128 / 32 = 4)

    std::vector<bool> fp32_dest_acc_ens = {false, true};

    for (bool fp32_dest_acc_en : fp32_dest_acc_ens) {
        bool this_passed =
            test_sdpa_sequential_phase1(devices_[0], q_chunk_tiles, k_chunk_tiles, inner_dim_tiles, fp32_dest_acc_en);

        if (!this_passed) {
            log_error(
                LogTest,
                "Full sequence test failed for q_chunk_tiles: {}, k_chunk_tiles: {}, "
                "inner_dim_tiles: {}, fp32_dest_acc_en: {}",
                q_chunk_tiles,
                k_chunk_tiles,
                inner_dim_tiles,
                fp32_dest_acc_en);
        }
        pass &= this_passed;
    }

    ASSERT_TRUE(pass);
}
