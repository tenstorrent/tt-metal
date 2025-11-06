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

// Create inputs for matmul: tensor_A [k_chunk_size*32 x head_dim*32],
// tensor_B [head_dim*32 x q_chunk_size*32], both in row-major layout.
static void create_matmul_inputs(
    uint32_t q_chunk_size,
    uint32_t k_chunk_size,
    uint32_t head_dim,
    std::vector<bfloat16>& tensor_A_rm,
    std::vector<bfloat16>& tensor_B_rm) {
    SHAPE a_shape = {1, 1, k_chunk_size * 32, head_dim * 32};
    SHAPE b_shape = {1, 1, head_dim * 32, q_chunk_size * 32};

    tt::deprecated::Tensor<bfloat16> a_tensor =
        tt::deprecated::initialize_tensor<bfloat16>(a_shape, tt::deprecated::Initialize::RANDOM, -1, 1, 0 /* seed */);
    tt::deprecated::Tensor<bfloat16> b_tensor =
        tt::deprecated::initialize_tensor<bfloat16>(b_shape, tt::deprecated::Initialize::RANDOM, -1, 1, 1 /* seed */);

    tensor_A_rm = a_tensor.get_values();
    tensor_B_rm = b_tensor.get_values();
}

// Golden reference: compute matmul(A, B) and reduce_max over dim=0 (rows) of the matmul output.
// A is [M x K], B is [K x N], C = A*B is [M x N]. reduce_max over dim=0 yields [1 x N].
static void golden_matmul_and_reduce_max(
    const std::vector<bfloat16>& tensor_A_rm,
    const std::vector<bfloat16>& tensor_B_rm,
    uint32_t q_chunk_size,
    uint32_t k_chunk_size,
    uint32_t head_dim,
    std::vector<bfloat16>& matmul_out_rm,
    std::vector<bfloat16>& max_out_rm) {
    const uint32_t M = k_chunk_size * 32;
    const uint32_t N = q_chunk_size * 32;
    const uint32_t K = head_dim * 32;

    matmul_out_rm.assign(M * N, static_cast<bfloat16>(0.0f));
    for (uint32_t i = 0; i < M; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; ++k) {
                acc += static_cast<float>(tensor_A_rm[i * K + k]) * static_cast<float>(tensor_B_rm[k * N + j]);
            }
            matmul_out_rm[i * N + j] = static_cast<bfloat16>(acc);
        }
    }

    max_out_rm.assign(N, static_cast<bfloat16>(0.0f));
    for (uint32_t j = 0; j < N; ++j) {
        float col_max = -std::numeric_limits<float>::infinity();
        for (uint32_t i = 0; i < M; ++i) {
            col_max = std::max(col_max, static_cast<float>(matmul_out_rm[i * N + j]));
        }
        max_out_rm[j] = static_cast<bfloat16>(col_max);
    }
}

static bool test_sdpa_reduce_c_transposed(
    tt_metal::IDevice* device, uint32_t q_chunk_size, uint32_t k_chunk_size, uint32_t head_dim, bool fp32_dest_acc_en) {
    bool pass = true;

    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    log_info(
        LogTest,
        "Running sdpa_reduce_c_transposed test with q_chunk_size: {}, k_chunk_size: {}, "
        "fp32_dest_acc_en: {}",
        q_chunk_size,
        k_chunk_size,
        fp32_dest_acc_en);

    // New test inputs per updated spec: build A [k_chunk_size*32 x head_dim*32]
    // and B [head_dim*32 x q_chunk_size*32], and compute golden matmul and reduce_max over dim=0.
    std::vector<bfloat16> tensor_A_rm;
    std::vector<bfloat16> tensor_B_rm;
    create_matmul_inputs(q_chunk_size, k_chunk_size, head_dim, tensor_A_rm, tensor_B_rm);

    std::vector<bfloat16> golden_matmul_rm;
    std::vector<bfloat16> golden_max_rm;
    golden_matmul_and_reduce_max(
        tensor_A_rm, tensor_B_rm, q_chunk_size, k_chunk_size, head_dim, golden_matmul_rm, golden_max_rm);

    // Set up device buffers for matmul and reduce_max
    const uint32_t M = k_chunk_size * 32;
    const uint32_t K = head_dim * 32;
    const uint32_t N = q_chunk_size * 32;

    auto cb_df = tt::DataFormat::Float16_b;
    auto cb_tile_size = tt::tile_size(cb_df);

    uint32_t k_in_num_tiles = k_chunk_size * head_dim;
    uint32_t q_in_num_tiles = head_dim * q_chunk_size;
    uint32_t mm_out_num_tiles = k_chunk_size * q_chunk_size;
    uint32_t max_out_num_tiles = q_chunk_size;

    // Program and core must be created before using globally allocated addresses in CBs
    tt_metal::Program program = tt_metal::CreateProgram();
    CoreCoord core = {0, 0};

    auto k_in_buffer_config = tt::tt_metal::ShardedBufferConfig{
        .device = device,
        .size = k_in_num_tiles * cb_tile_size,
        .page_size = cb_tile_size,
        .buffer_type = tt::tt_metal::BufferType::L1,
        .buffer_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = tt::tt_metal::ShardSpecBuffer(
            CoreRangeSet(std::set<CoreRange>({CoreRange(core, core)})),
            {k_chunk_size * tt::constants::TILE_HEIGHT, head_dim * tt::constants::TILE_WIDTH},
            tt::tt_metal::ShardOrientation::ROW_MAJOR,
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            {k_chunk_size, head_dim})};

    auto q_in_buffer_config = tt::tt_metal::ShardedBufferConfig{
        .device = device,
        .size = q_in_num_tiles * cb_tile_size,
        .page_size = cb_tile_size,
        .buffer_type = tt::tt_metal::BufferType::L1,
        .buffer_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = tt::tt_metal::ShardSpecBuffer(
            CoreRangeSet(std::set<CoreRange>({CoreRange(core, core)})),
            {head_dim * tt::constants::TILE_HEIGHT, q_chunk_size * tt::constants::TILE_WIDTH},
            tt::tt_metal::ShardOrientation::ROW_MAJOR,
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            {head_dim, q_chunk_size})};

    auto mm_output_buffer_config = tt::tt_metal::ShardedBufferConfig{
        .device = device,
        .size = mm_out_num_tiles * cb_tile_size,
        .page_size = cb_tile_size,
        .buffer_type = tt::tt_metal::BufferType::L1,
        .buffer_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = tt::tt_metal::ShardSpecBuffer(
            CoreRangeSet(std::set<CoreRange>({CoreRange(core, core)})),
            {k_chunk_size * tt::constants::TILE_HEIGHT, q_chunk_size * tt::constants::TILE_WIDTH},
            tt::tt_metal::ShardOrientation::ROW_MAJOR,
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            {k_chunk_size, q_chunk_size})};

    auto max_out_buffer_config = tt::tt_metal::ShardedBufferConfig{
        .device = device,
        .size = max_out_num_tiles * cb_tile_size,
        .page_size = cb_tile_size,
        .buffer_type = tt::tt_metal::BufferType::L1,
        .buffer_layout = tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = tt::tt_metal::ShardSpecBuffer(
            CoreRangeSet(std::set<CoreRange>({CoreRange(core, core)})),
            {tt::constants::TILE_HEIGHT, q_chunk_size * tt::constants::TILE_WIDTH},
            tt::tt_metal::ShardOrientation::ROW_MAJOR,
            {tt::constants::TILE_HEIGHT, tt::constants::TILE_WIDTH},
            {1, q_chunk_size})};

    auto one_tile_buffer_config_2 = tt::tt_metal::ShardedBufferConfig{
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

    auto k_in_buffer = CreateBuffer(k_in_buffer_config);
    auto q_in_buffer = CreateBuffer(q_in_buffer_config);
    auto mm_output_buffer = CreateBuffer(mm_output_buffer_config);
    auto max_out_buffer = CreateBuffer(max_out_buffer_config);
    auto one_tile_buffer = CreateBuffer(one_tile_buffer_config_2);

    // Host writes for inputs
    // k_in: tensor_A_rm, tilized
    {
        auto k_in_tilized = tilize_nfaces(tensor_A_rm, M, K);
        auto k_in_uint = pack_bfloat16_vec_into_uint32_vec(k_in_tilized);
        tt_metal::detail::WriteToBuffer(k_in_buffer, k_in_uint);
    }

    // q_in: transpose tiles before tilizing
    {
        std::vector<bfloat16> q_in_rm = tensor_B_rm;  // K x N
        transpose_tiles_inplace_row_major(q_in_rm, K, N);
        auto q_in_tilized = tilize_nfaces(q_in_rm, K, N);
        auto q_in_uint = pack_bfloat16_vec_into_uint32_vec(q_in_tilized);
        tt_metal::detail::WriteToBuffer(q_in_buffer, q_in_uint);
    }

    // one_tile_buffer: tile of ones
    {
        std::vector<bfloat16> ones_tile(
            tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH, static_cast<bfloat16>(1.0f));
        auto ones_uint = pack_bfloat16_vec_into_uint32_vec(ones_tile);
        tt_metal::detail::WriteToBuffer(one_tile_buffer, ones_uint);
    }

    // Create CircularBuffers for each buffer
    auto cb_k_in_id = tt::CBIndex::c_0;
    auto cb_k_in_config = tt::tt_metal::CircularBufferConfig(k_in_num_tiles * cb_tile_size, {{cb_k_in_id, cb_df}})
                              .set_page_size(cb_k_in_id, cb_tile_size)
                              .set_globally_allocated_address(*k_in_buffer);
    tt_metal::CreateCircularBuffer(program, core, cb_k_in_config);

    auto cb_q_in_id = tt::CBIndex::c_1;
    auto cb_q_in_config = tt::tt_metal::CircularBufferConfig(q_in_num_tiles * cb_tile_size, {{cb_q_in_id, cb_df}})
                              .set_page_size(cb_q_in_id, cb_tile_size)
                              .set_globally_allocated_address(*q_in_buffer);
    tt_metal::CreateCircularBuffer(program, core, cb_q_in_config);

    auto cb_mm_out_id = tt::CBIndex::c_2;
    auto cb_mm_out_config = tt::tt_metal::CircularBufferConfig(mm_out_num_tiles * cb_tile_size, {{cb_mm_out_id, cb_df}})
                                .set_page_size(cb_mm_out_id, cb_tile_size)
                                .set_globally_allocated_address(*mm_output_buffer);
    tt_metal::CreateCircularBuffer(program, core, cb_mm_out_config);

    auto cb_max_out_id = tt::CBIndex::c_3;
    auto cb_max_out_config =
        tt::tt_metal::CircularBufferConfig(max_out_num_tiles * cb_tile_size, {{cb_max_out_id, cb_df}})
            .set_page_size(cb_max_out_id, cb_tile_size)
            .set_globally_allocated_address(*max_out_buffer);
    tt_metal::CreateCircularBuffer(program, core, cb_max_out_config);

    auto cb_identity_scale_id = tt::CBIndex::c_4;
    auto cb_identity_scale_config =
        tt::tt_metal::CircularBufferConfig(1 * cb_tile_size, {{cb_identity_scale_id, cb_df}})
            .set_page_size(cb_identity_scale_id, cb_tile_size)
            .set_globally_allocated_address(*one_tile_buffer);
    tt_metal::CreateCircularBuffer(program, core, cb_identity_scale_config);

    /**
     * Determine matmul blocking information
     */
    const uint32_t dst_size = fp32_dest_acc_en ? 4 : 8;
    uint32_t out_subblock_h = std::min(k_chunk_size, dst_size);
    uint32_t out_subblock_w =
        (out_subblock_h == k_chunk_size) ? (std::min(q_chunk_size, dst_size / out_subblock_h)) : 1;

    if (out_subblock_h == dst_size && out_subblock_w == 1 && k_chunk_size % 2 == 0 && q_chunk_size % 2 == 0) {
        // Hacky, try to get 4x2 output subblock if possible to optimize matmul util.
        out_subblock_h = out_subblock_h / 2;
        out_subblock_w = 2;
    }

    const uint32_t in0_num_subblocks = k_chunk_size / out_subblock_h;
    const uint32_t in1_num_subblocks = q_chunk_size / out_subblock_w;

    log_info(
        tt::LogTest,
        "subblock_h: {}, subblock_w: {}, in0_num_subblocks: {}, in1_num_subblocks: {}",
        out_subblock_h,
        out_subblock_w,
        in0_num_subblocks,
        in1_num_subblocks);

    std::vector<uint32_t> compute_kernel_args = {
        cb_k_in_id,
        cb_q_in_id,
        cb_mm_out_id,
        cb_max_out_id,
        cb_identity_scale_id,
        k_chunk_size,
        q_chunk_size,
        head_dim,
        in0_num_subblocks,
        in1_num_subblocks,
        out_subblock_h,
        out_subblock_w};

    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/sdpa/matmul_reduce_transposed/compute.cpp",
        core,
        tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args});

    tt_metal::detail::LaunchProgram(device, program, true);

    std::vector<uint32_t> max_out_vec;
    tt_metal::detail::ReadFromBuffer(max_out_buffer, max_out_vec);
    auto max_out_bfp16 = unpack_uint32_vec_into_bfloat16_vec(max_out_vec);
    auto max_out_rm = untilize_nfaces(max_out_bfp16, 32, N);

    std::vector<uint32_t> mm_out_vec;
    tt_metal::detail::ReadFromBuffer(mm_output_buffer, mm_out_vec);
    auto mm_out_bfp16 = unpack_uint32_vec_into_bfloat16_vec(mm_out_vec);
    auto mm_out_rm = untilize_nfaces(mm_out_bfp16, M, N);

    // Correctness: compare device outputs with golden
    // 1) Matmul output full MxN
    {
        float mse_threshold = 0.003f;  // Determined empirically
        float mm_mse = 0.0f;
        const size_t elems = static_cast<size_t>(M) * static_cast<size_t>(N);
        for (size_t idx = 0; idx < elems; ++idx) {
            float a = static_cast<float>(mm_out_rm[idx]);
            float b = static_cast<float>(golden_matmul_rm[idx]);
            float d = a - b;
            mm_mse += d * d;
        }
        mm_mse /= static_cast<float>(elems);
        if (mm_mse > mse_threshold) {
            log_error(LogTest, "Matmul output MSE: {} > {}", mm_mse, mse_threshold);
            pass = false;
        }
    }

    // 2) Reduce-max output: compare the first row (row 0) of 32xN matrix to golden 1xN
    {
        float mse_threshold = 0.02f;
        float max_mse = 0.0f;
        for (uint32_t j = 0; j < N; ++j) {
            float a = static_cast<float>(max_out_rm[0 * N + j]);
            float b = static_cast<float>(golden_max_rm[j]);
            float d = a - b;
            max_mse += d * d;
        }
        max_mse /= static_cast<float>(N);
        if (max_mse > mse_threshold) {
            log_error(LogTest, "Reduce-max output MSE: {} > {}", max_mse, mse_threshold);
            pass = false;
        }
    }

    return pass;
}

/**
 * main
 *
 * This test drives a fused matmul + reduce_max pipeline on device and
 * verifies correctness against host-computed goldens.
 *
 * Operation under test
 * - Inputs:
 *   - tensor_A (aka k_in): shape M x K where M = k_chunk_size*32 and K = head_dim*32
 *   - tensor_B (aka q_in): shape K x N where N = q_chunk_size*32
 * - Device computes:
 *   1) matmul: C = A @ B, C shape is M x N
 *   2) reduce_max over dim=0 (rows) of C: max over M for each of the N columns
 *      producing a single row of N maxima.
 *
 * Tiling and memory layout
 * - All tensors are stored/streamed in tile format with fixed tile size 32x32.
 * - Host-side row-major buffers are converted to tile layout via tilize, and
 *   device results are converted back with untilize for validation.
 * - Special handling for q_in (tensor_B): each 32x32 tile of the KxN matrix is
 *   transposed in-place in row-major order BEFORE tilizing. In the device kernel,
 *   matmul_block is called with `transpose=true` to transpose RHS-tiles back.
 *
 * Buffers and CBs
 * - Inputs: k_in (A), q_in (B transposed per-tile), one_tile_buffer (tile of ones)
 * - Outputs: mm_output (C), max_out (reduce_max result)
 * - Each buffer has a corresponding CircularBuffer (CB) carrying bfloat16 tiles.
 *
 * Validation
 * - Host computes golden C and golden reduce_max using bfloat16 accumulations
 *   on the host vectors (converted via float for math, then cast back).
 * - The test compares:
 *   - Full C (MSE over all M*N elements) vs golden C
 *   - Reduce-max output: compares the first row of the device 32xN stats tensor
 *     with the 1xN golden max vector (the kernel writes stats in a 32xN layout,
 *     with maxima stored in row 0).
 */
int main(int argc, char** argv) {
    bool pass = true;
    char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
    putenv(env);

    int device_id = 0;
    tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);

    /**
     * Parameters to sweep over for correctness.
     */
    // sizes are in terms of tiles (32x32)
    std::vector<uint32_t> q_chunk_sizes = {1, 2, 4, 8};
    std::vector<uint32_t> k_chunk_sizes = {1, 2, 4, 8, 16};
    std::vector<uint32_t> head_dim_sizes = {1, 2, 4, 8};
    // Excluding fp32_dest_acc_en since SFPU reduce_max overlap will initially only support bf16 dst
    std::vector<bool> fp32_dest_acc_ens = {false};
    // Excluding do_eltwise since SFPU reduce_max overlap will not include eltwise max
    std::vector<bool> do_eltwise = {false};

    /**
     * These parameters are the same as the SDPA sprint-2 perfomance test parameters.
     * Uncomment to measure perf of the test we care most about.
     */
    // std::vector<uint32_t> q_chunk_sizes = {8};
    // std::vector<uint32_t> k_chunk_sizes = {16};
    // std::vector<uint32_t> head_dim_sizes = {4};
    // std::vector<bool> fp32_dest_acc_ens = {false};
    // std::vector<bool> do_eltwise = {false};

    for (uint32_t q_chunk_size : q_chunk_sizes) {
        for (uint32_t k_chunk_size : k_chunk_sizes) {
            for (uint32_t head_dim : head_dim_sizes) {
                for (bool fp32_dest_acc_en : fp32_dest_acc_ens) {
                    bool this_passed =
                        test_sdpa_reduce_c_transposed(device, q_chunk_size, k_chunk_size, head_dim, fp32_dest_acc_en);
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
